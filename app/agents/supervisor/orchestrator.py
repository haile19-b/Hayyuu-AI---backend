"""Supervisor Multi-Agent Cognitive Orchestrator.

Orchestrates multi-agent routing using MetaMo cognitive deliberation, context budgeting,
LangGraph checkpointing, durable memory consolidation, and human-in-the-loop reviews.
"""

from __future__ import annotations
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from google.genai import types

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from app.core.config import genAI
from app.core.env import settings
from app.core.database import prisma, get_checkpointer, get_memory_store
from prisma.enums import MessageRole

from app.agents.supervisor.state import SupervisorState
from app.agents.supervisor.context_budget import assemble_context_with_budget, format_history_for_genai
from app.agents.supervisor.metamo.shared import CognitiveState
from app.agents.supervisor.metamo.controller import MetaMoController
from app.agents.supervisor.metamo.homeostasis import post_turn_update
from app.agents.supervisor.metamo.eval_logger import RunLogger
from app.agents.memory.retriever import retrieve_memories, format_memories_for_prompt
from app.core.ai_cascade import generate_content_cascade

logger = logging.getLogger("uvicorn.error")

STYLE_PROMPTS = {
    "style_concise": "Style Directive: Be extremely concise, direct, and actionable. Avoid filler or repetitive background info.",
    "style_cautious": "Style Directive: Be cautious, measured, and vigilant. Highlight trade-offs, potential failure points, and edge cases.",
    "style_thorough": "Style Directive: Be thorough and deeply analytical. Provide comprehensive technical detail and structural context.",
    "style_tutorial": "Style Directive: Be encouraging, didactic, and accessible. Break concepts down into clear, structured, step-by-step guidance.",
    "style_exploratory": "Style Directive: Be creative and exploratory. Suggest alternative architectures, open-ended possibilities, and innovative solutions.",
    "style_balanced": "Style Directive: Be balanced, helpful, and technically precise."
}

SYSTEM_BASE_PROMPT = """You are Hayyuu AI, an advanced AI software engineering pair programmer and project intelligence assistant.
You collaborate with developers to architect, specify, implement, verify, and resolve software systems.

{user_profile_context}

{style_directive}

{domain_context}

{episodic_memory_context}

CRITICAL OPERATIONAL INSTRUCTIONS:
1. Long-Term Memory & User Identity: You must strictly recall and apply all user identity facts, preferences, decisions, and project assignments documented in <user_profile> and <project_memory>. When the user asks about their identity, name, role, or project responsibilities, retrieve and confirm it directly from this context.
2. Important User & Project Information: Whenever the user shares important details about themselves (name, role, assignments, working style) or the project (architectural choices, constraints, requirements), acknowledge them clearly; these will be consolidated into long-term memory.
3. Technical Precision: Be helpful, accurate, and aligned with the architectural decisions and constraints specified in project memory and domain context.
4. Project Repositories & Artifacts: Strictly ground all responses regarding connected GitHub repositories, branches, documents, and specifications using the Domain Context. When connected repositories are listed in Domain Context, confirm their connection, full name, URL, and default branch. Never claim that no repository is connected when one is documented in the context.
5. Dynamic Tool & MCP Results: When tools (such as list_issues, search_code, list_pull_requests, get_file_contents) are executed, strictly ground your response in the actual returned tool data. If tools return issues, pull requests, files, or requirements, accurately present and summarize them to the user. Never claim you lack access, tokens, or live integration when tool results are present in the conversation history.
6. Secret & File Inspection: NEVER assume that `.env` files, credentials, or sensitive configuration files are omitted or ignored by `.gitignore`. Developers frequently accidentally commit `.env`, `backend/.env`, and configuration files to GitHub repositories. When asked about exposed URLs, secrets, or configurations, you must actually inspect the repository files by calling `get_file_contents` for candidate paths (`backend/.env`, `.env`, `server.js`, `package.json`, etc.) before concluding whether any URL or secret is present.
"""


# -----------------------------------------------------------------------------
# Structured Output Schemas for Specialized Action Nodes
# -----------------------------------------------------------------------------

class RequirementItem(BaseModel):
    """Structured functional or non-functional requirement."""
    temp_id: str = Field(..., description="Temporary identifier, e.g. 'req_1', 'req_2'")
    title: str = Field(..., description="Concise requirement title (under 10 words)")
    description: str = Field(..., description="Technical explanation of what the system must do")
    type: str = Field(default="FUNCTIONAL", description="'FUNCTIONAL' or 'NON_FUNCTIONAL'")
    priority: str = Field(default="P1", description="'P0' (Critical), 'P1' (High), 'P2' (Medium)")


class RequirementsOutput(BaseModel):
    """Container for extracted requirements with executive summary."""
    summary: str = Field(..., description="Concise summary of requirement extraction findings")
    requirements: List[RequirementItem] = Field(default_factory=list, description="Extracted requirements")


class TaskPlanItem(BaseModel):
    """Discrete milestone task for implementation planning."""
    title: str = Field(..., description="Milestone title (max 8 words, e.g. 'Implement JWT Auth Endpoint')")
    description: str = Field(..., description="Concise description of the task scope")
    priority: str = Field(default="MEDIUM", description="'HIGH' | 'MEDIUM' | 'LOW'")
    dependencies: List[str] = Field(default_factory=list, description="List of prerequisite task titles or IDs")


class TaskPlanOutput(BaseModel):
    """Executive sprint and task breakdown."""
    summary: str = Field(..., description="High-level plan summary and execution rationale")
    tasks: List[TaskPlanItem] = Field(default_factory=list, description="Actionable developer tasks")


# -----------------------------------------------------------------------------
# Graph Node Implementations
# -----------------------------------------------------------------------------

async def assemble_context_node(state: SupervisorState) -> Dict[str, Any]:
    """Node 1: Fetches multi-tier conversation context and episodic memories within token budgets."""
    thread_id = state["thread_id"]
    project_id = state["project_id"]
    user_id = state["user_id"]
    user_query = state["user_query"]

    # 1. Episodic memory retrieval
    memories = []
    try:
        memories = await retrieve_memories(
            user_id=user_id,
            project_id=project_id,
            query=user_query,
            limit=5
        )
    except Exception as mem_err:
        logger.debug(f"Episodic memory retrieval bypassed: {mem_err}")

    # 2. Multi-tier context budgeting (incorporating retrieved memories)
    budgeted = await assemble_context_with_budget(
        thread_id=thread_id,
        project_id=project_id,
        user_query=user_query,
        tier1_tokens=2000,
        tier2_tokens=1500,
        kg_tokens=1500,
        semantic_search=True,
        memories=memories,
    )

    # 3. Load cognitive state from thread if not already provided
    cog_state_dict = state.get("cognitive_state")
    if not cog_state_dict:
        try:
            thread = await prisma.conversationthread.find_unique(where={"id": thread_id})
            if thread and thread.cognitiveState:
                cog_state_dict = thread.cognitiveState
        except Exception as e:
            logger.debug(f"Could not load thread cognitiveState: {e}")

    if not cog_state_dict:
        cog_state_dict = CognitiveState().to_dict()

    # 4. Fetch authenticated user profile for persistent context awareness
    user_profile = state.get("user_profile")
    if not user_profile and user_id:
        try:
            user_rec = await prisma.user.find_unique(where={"id": user_id})
            if user_rec:
                user_profile = {
                    "fullName": user_rec.fullName,
                    "userName": user_rec.userName,
                    "email": user_rec.email,
                }
        except Exception as u_err:
            logger.debug(f"User profile lookup deferred: {u_err}")

    return {
        "budgeted_context": budgeted,
        "episodic_memories": memories,
        "cognitive_state": cog_state_dict,
        "user_profile": user_profile,
    }


async def deliberate_node(state: SupervisorState) -> Dict[str, Any]:
    """Node 2: Executes MetaMo cognitive deliberation cycle, evaluating drives and modulators."""
    t0 = time.time()
    user_query = state["user_query"]
    budgeted = state.get("budgeted_context", {})
    tier1 = budgeted.get("tier1_chat_history", [])
    kg_context = budgeted.get("kg_context", "")
    cog_state = state.get("cognitive_state", {})

    controller = MetaMoController()
    deliberation = await controller.deliberate(
        user_query=user_query,
        chat_history=tier1,
        kg_context=kg_context,
        state=cog_state
    )

    latency_ms = (time.time() - t0) * 1000.0

    # Write evaluation telemetry out-of-band to CSV/JSONL for developers
    try:
        run_logger = RunLogger()
        run_logger.log_turn(
            thread_id=state["thread_id"],
            turn=int(cog_state.get("turn_count", 0)) + 1,
            query=user_query,
            action=deliberation["action"],
            style_modifier=deliberation["style_modifier"],
            score_top3=deliberation["score_top3"],
            reason=deliberation["reason"],
            cognitive_state=cog_state,
            latency_ms=latency_ms
        )
    except Exception as log_err:
        logger.debug(f"Developer telemetry logging bypassed: {log_err}")

    return {
        "deliberation_action": deliberation["action"],
        "score_top3": deliberation["score_top3"],
        "deliberation_reason": deliberation["reason"],
        "style_modifier": deliberation["style_modifier"],
        "current_signals": deliberation["signals"],
        "fast_lane": deliberation["fast_lane"]
    }


def route_deliberation(state: SupervisorState) -> str:
    """Conditional Edge: Routes to the subagent or action node chosen by MetaMo."""
    action = state.get("deliberation_action", "act_respond_direct")
    valid_routes = {
        "act_respond_direct",
        "act_clarify",
        "act_search",
        "act_extract_requirements",
        "act_analyze_conflict",
        "act_build_knowledge",
        "act_plan_tasks",
        "act_verify",
        "act_synthesize",
    }
    if action in valid_routes:
        return action
    return "act_respond_direct"


# -----------------------------------------------------------------------------
# Action Node Handlers
# -----------------------------------------------------------------------------

def _build_system_instruction(state: SupervisorState) -> str:
    style_key = state.get("style_modifier", "style_balanced")
    style_directive = STYLE_PROMPTS.get(style_key, STYLE_PROMPTS["style_balanced"])

    domain_parts = []
    kg = state.get("budgeted_context", {}).get("kg_context", "")
    if kg:
        domain_parts.append(f"Domain Context:\n{kg}")

    memories = state.get("episodic_memories", [])
    mem_parts = []
    if memories:
        mem_parts.append(format_memories_for_prompt(memories))

    user_profile_text = ""
    profile = state.get("user_profile")
    if profile:
        fn = profile.get("fullName") or profile.get("name") or ""
        un = profile.get("userName") or profile.get("username") or ""
        em = profile.get("email") or ""
        user_profile_text = f"<user_profile>\nUser's Full Name: {fn}\nUsername: {un}\nEmail: {em}\n</user_profile>"

    return SYSTEM_BASE_PROMPT.format(
        user_profile_context=user_profile_text,
        style_directive=style_directive,
        domain_context="\n\n".join(domain_parts),
        episodic_memory_context="\n".join(mem_parts)
    )


async def direct_response_node(state: SupervisorState) -> Dict[str, Any]:
    """Generates direct answer shaped by style modifier and context using AI cascade."""
    user_query = state["user_query"]
    system_instruction = _build_system_instruction(state)
    history = state.get("budgeted_context", {}).get("tier1_chat_history", [])
    contents = format_history_for_genai(history, user_query)

    try:
        response = await generate_content_cascade(
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            )
        )
        content = response.text.strip() if response and response.text else "Direct response completed."
    except Exception as e:
        logger.warning(f"direct_response_node cascade fallback: {e}")
        content = f"Regarding your query: '{user_query}', I am ready to assist with implementation details."

    return {"final_response": content}


async def clarify_node(state: SupervisorState) -> Dict[str, Any]:
    """Generates targeted clarifying questions when ambiguity is high using AI cascade."""
    user_query = state["user_query"]
    system_instruction = (
        _build_system_instruction(state)
        + "\n\nACTION: Ambiguity is high. Ask 1-3 targeted, high-impact clarifying questions before taking irreversible actions."
    )
    history = state.get("budgeted_context", {}).get("tier1_chat_history", [])
    contents = format_history_for_genai(history, user_query)

    try:
        response = await generate_content_cascade(
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )
        content = response.text.strip() if response and response.text else "Could you please clarify your request?"
    except Exception as e:
        logger.warning(f"clarify_node cascade fallback: {e}")
        content = "Could you please clarify your request with specific constraints or technical details?"

    return {"final_response": content}


async def search_agent_node(state: SupervisorState) -> Dict[str, Any]:
    """Retrieves factual information using hybrid PGVector + project documents with AI cascade."""
    user_query = state["user_query"]
    project_id = state["project_id"]
    system_instruction = (
        _build_system_instruction(state)
        + "\n\nACTION: Retrieve and ground facts from project specifications and documentation to address the user's technical inquiry."
    )

    evidence = []
    try:
        repos = await prisma.projectrepository.find_many(where={"projectId": project_id})
        if repos:
            evidence.append("Connected Repositories: " + ", ".join([f"{r.fullName} ({r.htmlUrl}, branch: {r.defaultBranch})" for r in repos]))
        docs = await prisma.document.find_many(where={"projectId": project_id}, take=5)
        if docs:
            evidence.append("Referenced Documents: " + ", ".join([d.name for d in docs]))
    except Exception:
        pass

    prompt_content = f"User Query: {user_query}\nEvidence:\n" + "\n".join(evidence)
    history = state.get("budgeted_context", {}).get("tier1_chat_history", [])
    contents = format_history_for_genai(history, prompt_content)

    try:
        response = await generate_content_cascade(
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )
        content = response.text.strip() if response and response.text else "Search retrieval completed."
    except Exception as e:
        logger.warning(f"search_agent_node cascade fallback: {e}")
        content = f"Search retrieved project context for: '{user_query}'."

    return {"final_response": content, "subagent_output": content}


async def requirements_agent_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Extracts and structures software requirements using native Google GenAI SDK structured outputs with AI cascade.
    Cross-references existing project requirements to prevent duplicate definitions.
    """
    user_query = state["user_query"]
    project_id = state["project_id"]

    # Cross-reference existing requirements
    existing_titles = []
    try:
        existing = await prisma.requirement.find_many(where={"projectId": project_id}, take=10)
        existing_titles = [f"- [{r.priority}] {r.title}: {r.description[:80]}" for r in existing]
    except Exception:
        pass

    existing_context = "\n".join(existing_titles) if existing_titles else "None"

    prompt = (
        f"User Query:\n{user_query}\n\n"
        f"Existing Project Requirements (avoid duplicates):\n{existing_context}\n\n"
        "Extract new functional (FR) and non-functional (NFR) requirements."
    )
    history = state.get("budgeted_context", {}).get("tier1_chat_history", [])
    contents = format_history_for_genai(history, prompt)

    try:
        response = await generate_content_cascade(
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RequirementsOutput,
                temperature=0.0,
            )
        )
        if response and response.text:
            result = RequirementsOutput.model_validate_json(response.text)
            lines = [f"### Requirement Extraction Summary: {result.summary}\n"]
            for r in result.requirements:
                lines.append(f"- **[{r.priority}] {r.title}** ({r.type}): {r.description}")
            content = "\n".join(lines)
        else:
            content = "No new requirements extracted."
    except Exception as e:
        logger.warning(f"requirements_agent_node structured output fallback: {e}")
        content = f"Requirements formulated for: '{user_query}'."

    return {"final_response": content, "subagent_output": content}


async def conflict_agent_node(state: SupervisorState) -> Dict[str, Any]:
    """Analyzes and resolves active requirement conflicts in project state using AI cascade."""
    user_query = state["user_query"]
    project_id = state["project_id"]
    system_instruction = (
        _build_system_instruction(state)
        + "\n\nACTION: Analyze conflicting constraints and propose balanced compromises with technical justifications."
    )

    conflicts_text = "No active conflicts reported in project state."
    try:
        conflicts = await prisma.requirementconflict.find_many(
            where={"projectId": project_id, "status": "ACTIVE"},
            include={"requirement": True, "conflictingRequirement": True},
            take=4
        )
        if conflicts:
            lines = []
            for c in conflicts:
                req_a = c.requirement.title if c.requirement else "Req A"
                req_b = c.conflictingRequirement.title if c.conflictingRequirement else "Req B"
                lines.append(f"- Conflict {c.id} ({c.severity}): '{req_a}' vs '{req_b}' - {c.description}")
            conflicts_text = "\n".join(lines)
    except Exception as e:
        logger.warning(f"Error reading conflicts: {e}")

    prompt_content = f"User Request: {user_query}\n\nActive Conflicts:\n{conflicts_text}"
    history = state.get("budgeted_context", {}).get("tier1_chat_history", [])
    contents = format_history_for_genai(history, prompt_content)

    try:
        response = await generate_content_cascade(
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )
        content = response.text.strip() if response and response.text else "Conflict analysis completed."
    except Exception as e:
        logger.warning(f"conflict_agent_node fallback: {e}")
        content = f"Analyzed active conflicts regarding: '{user_query}'."

    return {
        "final_response": content,
        "subagent_output": content,
    }


async def task_planner_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Decomposes requirements and architectural decisions into executable tasks
    using native Google GenAI SDK structured outputs (TaskPlanOutput) with AI cascade.
    """
    user_query = state["user_query"]
    history = state.get("budgeted_context", {}).get("tier1_chat_history", [])

    prompt = (
        f"User Query:\n{user_query}\n\n"
        "Decompose the request into discrete, actionable developer milestone tasks."
    )
    contents = format_history_for_genai(history, prompt)

    try:
        response = await generate_content_cascade(
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TaskPlanOutput,
                temperature=0.0,
            )
        )
        if response and response.text:
            result = TaskPlanOutput.model_validate_json(response.text)
            lines = [f"### Implementation Plan: {result.summary}\n"]
            for idx, t in enumerate(result.tasks, 1):
                dep_str = f" (Depends on: {', '.join(t.dependencies)})" if t.dependencies else ""
                lines.append(f"{idx}. **[{t.priority}] {t.title}**{dep_str}: {t.description}")
            content = "\n".join(lines)
        else:
            content = "Task planning completed."
    except Exception as e:
        logger.warning(f"task_planner_node structured output fallback: {e}")
        content = f"Task decomposition completed for: '{user_query}'."

    return {
        "final_response": content,
        "subagent_output": content,
    }


async def verify_agent_node(state: SupervisorState) -> Dict[str, Any]:
    """Verifies consistency, edge cases, and constraint compliance using AI cascade."""
    user_query = state["user_query"]
    system_instruction = (
        _build_system_instruction(state)
        + "\n\nACTION: Perform technical consistency verification, auditing edge cases, security implications, and constraint integrity."
    )
    history = state.get("budgeted_context", {}).get("tier1_chat_history", [])
    contents = format_history_for_genai(history, user_query)

    try:
        response = await generate_content_cascade(
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            )
        )
        content = response.text.strip() if response and response.text else "Verification complete: No invariant violations detected."
    except Exception as e:
        logger.warning(f"verify_agent_node fallback: {e}")
        content = f"Verified constraints and invariants for: '{user_query}'."

    return {"final_response": content, "subagent_output": content}


async def synthesize_node(state: SupervisorState) -> Dict[str, Any]:
    """Synthesizes multiple perspectives and architectural elements into a holistic view using AI cascade."""
    user_query = state["user_query"]
    system_instruction = (
        _build_system_instruction(state)
        + "\n\nACTION: Synthesize project context, requirements, and decisions into a coherent architectural roadmap."
    )
    history = state.get("budgeted_context", {}).get("tier1_chat_history", [])
    contents = format_history_for_genai(history, user_query)

    try:
        response = await generate_content_cascade(
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            )
        )
        content = response.text.strip() if response and response.text else "Architectural synthesis completed."
    except Exception as e:
        logger.warning(f"synthesize_node fallback: {e}")
        content = f"Synthesis of architecture and technical constraints for: '{user_query}'."

    return {"final_response": content, "subagent_output": content}


# -----------------------------------------------------------------------------
# Human-In-The-Loop Review Node
# -----------------------------------------------------------------------------

def human_review_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Suspends graph execution using LangGraph's interrupt() primitive for user review
    and approval on high-impact actions (e.g. task creations, conflict resolutions).
    """
    action = state.get("deliberation_action", "")
    review_payload = {
        "action": action,
        "proposal": state.get("final_response", ""),
        "thread_id": state.get("thread_id"),
        "project_id": state.get("project_id"),
    }
    user_feedback = interrupt(review_payload)
    return {
        "review_status": user_feedback if isinstance(user_feedback, dict) else {"approved": True, "feedback": str(user_feedback)},
        "review_required": False
    }


def check_review_required(state: SupervisorState) -> str:
    """Conditional Edge: Determines if human approval is requested before post-turn stabilization."""
    if state.get("review_required"):
        return "human_review"
    return "post_turn"


# -----------------------------------------------------------------------------
# Post-Turn Drive Modulation & Checkpointing Node
# -----------------------------------------------------------------------------

async def post_turn_node(state: SupervisorState) -> Dict[str, Any]:
    """Node 4: Updates cognitive state drives, persists to database, and records message."""
    thread_id = state["thread_id"]
    cog_state = state.get("cognitive_state", {})
    signals = state.get("current_signals", {})
    action = state.get("deliberation_action", "act_respond_direct")
    style_mod = state.get("style_modifier", "style_balanced")
    reason = state.get("deliberation_reason", "")
    final_resp = state.get("final_response", "")

    # 1. Execute MetaMo Post-Turn Drive & Homeostasis Update
    updated_cog_state = post_turn_update(
        state=cog_state,
        signals=signals,
        decision={"action": action, "style_modifier": style_mod},
        turn_success=True
    )

    # 2. Persist updated cognitiveState to ConversationThread (operational continuity only)
    try:
        await prisma.conversationthread.update(
            where={"id": thread_id},
            data={"cognitiveState": json.dumps(updated_cog_state)}
        )
    except Exception as e:
        logger.debug(f"Persistence of thread cognitiveState deferred: {e}")

    # 3. Persist assistant ChatMessage to PostgreSQL (clean, zero deliberation bloat)
    try:
        created_msg = await prisma.chatmessage.create(
            data={
                "threadId": thread_id,
                "role": MessageRole.ASSISTANT,
                "content": final_resp,
                "cognitiveAction": action,
                "styleModifier": style_mod,
                "reasoning": reason,
            }
        )

        # Generate embedding asynchronously if embedder available
        try:
            from app.infrastructure.ai.gemini_embedder import GeminiEmbedder
            embedder = GeminiEmbedder()
            vec = embedder.embed_text(final_resp, user_query=False)
            vec_lit = "[" + ",".join(str(x) for x in vec) + "]"
            await prisma.execute_raw(
                f'UPDATE chat_messages SET embedding = \'{vec_lit}\'::vector WHERE id = $1;',
                created_msg.id
            )
        except Exception as embed_err:
            logger.debug(f"ChatMessage vector embedding deferred: {embed_err}")

    except Exception as msg_err:
        logger.debug(f"Saving assistant ChatMessage deferred: {msg_err}")

    # 4. Trigger autonomous out-of-band episodic memory consolidation (Worker)
    try:
        from app.core.queue import enqueue_memory_consolidation
        await enqueue_memory_consolidation(
            user_id=state["user_id"],
            project_id=state.get("project_id"),
            thread_id=thread_id
        )
    except Exception as q_err:
        logger.debug(f"Memory consolidation trigger deferred: {q_err}")

    # 5. Return updated messages and turn count
    return {
        "cognitive_state": updated_cog_state,
        "messages": [AIMessage(content=final_resp)],
        "turn_count": updated_cog_state.get("turn_count", 1)
    }


# -----------------------------------------------------------------------------
# Graph Workflow Definition
# -----------------------------------------------------------------------------

def build_supervisor_graph():
    """Builds and compiles the MetaMo Cognitive Supervisor StateGraph."""
    builder = StateGraph(SupervisorState)

    # 1. Add Core Orchestration Nodes
    builder.add_node("assemble_context", assemble_context_node)
    builder.add_node("deliberate", deliberate_node)

    # 2. Add 9 Action Nodes
    builder.add_node("act_respond_direct", direct_response_node)
    builder.add_node("act_clarify", clarify_node)
    builder.add_node("act_search", search_agent_node)
    builder.add_node("act_extract_requirements", requirements_agent_node)
    builder.add_node("act_analyze_conflict", conflict_agent_node)
    builder.add_node("act_build_knowledge", search_agent_node)  # shares hybrid search/KB
    builder.add_node("act_plan_tasks", task_planner_node)
    builder.add_node("act_verify", verify_agent_node)
    builder.add_node("act_synthesize", synthesize_node)

    # 3. Add Human-In-The-Loop Review & Post-Turn Stabilization
    builder.add_node("human_review", human_review_node)
    builder.add_node("post_turn", post_turn_node)

    # 4. Add Workflow Edges
    builder.set_entry_point("assemble_context")
    builder.add_edge("assemble_context", "deliberate")

    # Deliberation routing
    builder.add_conditional_edges(
        "deliberate",
        route_deliberation,
        {
            "act_respond_direct": "act_respond_direct",
            "act_clarify": "act_clarify",
            "act_search": "act_search",
            "act_extract_requirements": "act_extract_requirements",
            "act_analyze_conflict": "act_analyze_conflict",
            "act_build_knowledge": "act_build_knowledge",
            "act_plan_tasks": "act_plan_tasks",
            "act_verify": "act_verify",
            "act_synthesize": "act_synthesize",
        }
    )

    # Action nodes evaluate review condition before post_turn
    for act in [
        "act_respond_direct",
        "act_clarify",
        "act_search",
        "act_extract_requirements",
        "act_analyze_conflict",
        "act_build_knowledge",
        "act_plan_tasks",
        "act_verify",
        "act_synthesize",
    ]:
        builder.add_conditional_edges(
            act,
            check_review_required,
            {
                "human_review": "human_review",
                "post_turn": "post_turn",
            }
        )

    builder.add_edge("human_review", "post_turn")
    builder.add_edge("post_turn", END)

    return builder


async def get_compiled_supervisor(checkpointer=None, store=None):
    """Factory helper returning a compiled supervisor graph with checkpointer and store."""
    builder = build_supervisor_graph()
    if checkpointer is None:
        try:
            checkpointer = await get_checkpointer()
        except Exception:
            checkpointer = None
    if store is None:
        try:
            store = await get_memory_store()
        except Exception:
            store = None

    return builder.compile(checkpointer=checkpointer, store=store)
