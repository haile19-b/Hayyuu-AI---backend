"""Node 2: Deliberate Node for Chat Agent.

Executes MetaMo cognitive deliberation cycle, evaluating drives, modulators,
and action routing. Builds dynamic system prompt and formats multi-turn history.
"""

import time
import json
import re
import logging
from typing import Dict, Any, List

from google.genai import types

from app.core.ai_cascade import generate_content_cascade
from app.infrastructure.mcp.client_manager import MCPClientManager
from app.agents.supervisor.metamo.controller import MetaMoController
from app.agents.supervisor.metamo.eval_logger import RunLogger
from app.agents.supervisor.orchestrator import STYLE_PROMPTS, SYSTEM_BASE_PROMPT
from app.agents.supervisor.context_budget import format_history_for_genai
from app.agents.memory.retriever import format_memories_for_prompt
from app.agents.chat_agent.state import ChatAgentState

logger = logging.getLogger("uvicorn.error")

TOOL_KEYWORDS = (
    "issue", "pr", "pull request", "repo", "repository", "github", "git",
    "branch", "commit", "code", "file", "diff", "task", "requirement",
    "conflict", "database", "neo4j", "subgraph"
)

PLANNER_SYSTEM_PROMPT = """You are Hayyuu AI's Autonomous Engineering Planner.
Analyze the user's inquiry, workspace domain context, and available tools.
Determine if external tools (GitHub, Neo4j, documents, tasks) are required to answer the query accurately.

CRITICAL ENGINEERING PLANNING RULES:
1. Secret, URL & Configuration Discovery:
   - When searching for exposed URLs, database connections, credentials, or environment secrets:
     DO NOT rely solely on `search_code`, because GitHub's code search API deliberately ignores and excludes `.env` files and dotfiles from its search index.
     Instead, plan to:
     a) Inspect repository structure or directory contents using `get_file_contents(path="")` to discover subdirectories (e.g. `backend/`, `server/`).
     b) Specifically inspect environment and config files (e.g. `backend/.env`, `.env`, `docker-compose.yaml`, `server.js`, `package.json`) using `get_file_contents`.
2. Issue & PR Inquiries:
   - Use `list_issues` (with state='all' or 'open') and `get_issue`.
3. Code Architecture & Implementation:
   - Combine `search_code` and `get_file_contents` to read actual file bodies.

If tools are NOT needed (e.g. conversational greetings, questions about user profile/name, or facts already fully detailed in Domain Context), output:
{
  "reasoning": "Direct conversational response",
  "needs_tools": false,
  "steps": []
}

If tools ARE needed, break down the investigation into 1 to 4 sequential, goal-oriented steps:
{
  "reasoning": "Explain what needs to be discovered or verified",
  "needs_tools": true,
  "steps": [
    {
      "id": "step_1",
      "description": "Specific sub-goal to achieve",
      "tool_hint": "name of the most suitable tool"
    }
  ]
}

Return strictly valid JSON.
"""


async def _generate_plan(
    user_query: str,
    domain_context: str,
    action_key: str
) -> Dict[str, Any]:
    """Generates an explicit investigation plan using the unified cascade engine."""
    q_lower = user_query.lower()
    has_keywords = any(kw in q_lower for kw in TOOL_KEYWORDS)
    is_action_tool = action_key in ("act_search", "act_verify", "act_synthesize", "act_analyze_conflict")

    if not has_keywords and not is_action_tool:
        return {"reasoning": "Conversational query", "needs_tools": False, "steps": []}

    tool_names = []
    try:
        mcp_mgr = MCPClientManager.get_instance()
        if mcp_mgr and mcp_mgr.available_tools:
            tool_names = list(mcp_mgr.available_tools.keys())
    except Exception:
        pass

    prompt_content = f"""User Query: "{user_query}"

{domain_context}

Available Tools:
{tool_names if tool_names else 'list_issues, get_issue, search_code, get_file_contents, list_project_repositories, list_project_tasks, list_project_documents'}
"""
    try:
        resp = await generate_content_cascade(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_content)])],
            config=types.GenerateContentConfig(
                system_instruction=PLANNER_SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json"
            ),
            timeout_seconds=20.0
        )
        if resp and resp.text:
            cleaned = resp.text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-z]*\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
            steps = parsed.get("steps", [])
            for i, step in enumerate(steps, 1):
                if not step.get("id"):
                    step["id"] = f"step_{i}"
                step["status"] = "pending"
                step["result_summary"] = None
            return {
                "reasoning": parsed.get("reasoning", ""),
                "needs_tools": bool(steps),
                "steps": steps
            }
    except Exception as plan_err:
        logger.warning(f"[ChatAgent: deliberate] Planning generation fallback: {plan_err}")

    # Fallback plan if keyword detected but LLM planning failed
    if has_keywords:
        default_tool = "list_issues" if "issue" in q_lower else ("search_code" if "code" in q_lower else None)
        return {
            "reasoning": "Keyword-triggered autonomous investigation",
            "needs_tools": True,
            "steps": [
                {
                    "id": "step_1",
                    "description": f"Investigate information for '{user_query}'",
                    "tool_hint": default_tool,
                    "status": "pending",
                    "result_summary": None
                }
            ]
        }

    return {"reasoning": "Direct response", "needs_tools": False, "steps": []}


async def deliberate_node(state: ChatAgentState) -> Dict[str, Any]:
    """Node 2: MetaMo deliberation, cognitive decision routing, and prompt assembly."""
    t0 = time.time()
    user_query = state["user_query"]
    thread_id = state["thread_id"]
    budgeted = state.get("budgeted_context", {})
    tier1 = budgeted.get("tier1_chat_history", [])
    kg_context = budgeted.get("kg_context", "")
    cog_state = state.get("cognitive_state", {})

    # 1. Execute MetaMo deliberation
    controller = MetaMoController()
    deliberation = await controller.deliberate(
        user_query=user_query,
        chat_history=tier1,
        kg_context=kg_context,
        state=cog_state
    )

    latency_ms = (time.time() - t0) * 1000.0

    # 2. Log evaluation telemetry out-of-band for developers
    try:
        run_logger = RunLogger()
        run_logger.log_turn(
            thread_id=thread_id,
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
        logger.debug(f"[ChatAgent: deliberate] Telemetry logging bypassed: {log_err}")

    # 3. Build dynamic action directive
    action_key = deliberation.get("action", "act_respond_direct")
    action_directive = ""
    if action_key == "act_clarify":
        action_directive = "\n\nACTION: Ambiguity is high. Ask 1-3 targeted, high-impact clarifying questions before taking irreversible actions."
    elif action_key == "act_search":
        action_directive = "\n\nACTION: Retrieve and ground facts from project specifications and documentation to address the user's technical inquiry."
    elif action_key in ("act_extract_requirements", "act_plan_tasks", "act_decompose"):
        action_directive = "\n\nACTION: Provide a structured breakdown of requirements or task plan for the engineering request."
    elif action_key == "act_verify":
        action_directive = "\n\nACTION: Perform rigorous verification and conflict detection across technical specifications."
    elif action_key == "act_synthesize":
        action_directive = "\n\nACTION: Synthesize complex requirements, architectural components, and data flows."

    # 4. Build user profile context
    user_profile = state.get("user_profile")
    user_profile_text = ""
    if user_profile:
        fn = user_profile.get("fullName") or ""
        un = user_profile.get("userName") or ""
        em = user_profile.get("email") or ""
        user_profile_text = f"<user_profile>\nUser's Full Name: {fn}\nUsername: {un}\nEmail: {em}\n</user_profile>"

    # 5. Format episodic memories
    memories = state.get("episodic_memories", [])
    mem_formatted = format_memories_for_prompt(memories) if memories else ""
    episodic_block = f"Long-Term User Preferences & Decisions:\n{mem_formatted}" if mem_formatted else ""

    # 6. Build dynamic system instruction with rich domain context (repos, docs, knowledge graph)
    domain_parts = []
    if kg_context:
        domain_parts.append(kg_context)

    connected_repos = state.get("connected_repositories", [])
    if connected_repos:
        repo_lines = ["Connected GitHub Repositories:"]
        for r in connected_repos:
            repo_lines.append(f"- Repository: {r.get('fullName') or r.get('name')} (URL: {r.get('htmlUrl')}, Default Branch: {r.get('defaultBranch', 'main')})")
        domain_parts.append("\n".join(repo_lines))

    project_docs = state.get("project_documents", [])
    if project_docs:
        doc_lines = ["Project Documents:"]
        for d in project_docs:
            doc_lines.append(f"- Document: {d.get('name')} ({d.get('fileType')}, Status: {d.get('status')})")
        domain_parts.append("\n".join(doc_lines))

    domain_context_str = ("Domain Context:\n" + "\n\n".join(domain_parts)) if domain_parts else ""

    style_key = deliberation.get("style_modifier", "style_balanced")
    style_directive = STYLE_PROMPTS.get(style_key, STYLE_PROMPTS["style_balanced"])

    system_instruction = SYSTEM_BASE_PROMPT.format(
        user_profile_context=user_profile_text,
        style_directive=style_directive,
        domain_context=domain_context_str,
        episodic_memory_context=episodic_block
    ) + action_directive

    # 7. Format multi-turn conversational history for Gemini
    gemini_contents = format_history_for_genai(tier1, user_query)

    # 8. Plan -> Do: Formulate explicit investigation plan
    plan_data = await _generate_plan(
        user_query=user_query,
        domain_context=domain_context_str,
        action_key=action_key
    )
    steps = plan_data.get("steps", [])
    needs_tools = plan_data.get("needs_tools", False) and len(steps) > 0

    return {
        "deliberation_action": deliberation["action"],
        "deliberation_reason": deliberation["reason"],
        "style_modifier": deliberation["style_modifier"],
        "score_top3": deliberation["score_top3"],
        "signals": deliberation["signals"],
        "fast_lane": deliberation["fast_lane"],
        "system_instruction": system_instruction,
        "gemini_contents": gemini_contents,
        "needs_tools": needs_tools,
        "investigation_plan": steps,
        "current_plan_step_idx": 0,
        "is_plan_complete": not needs_tools,
        "structured_evidence": "",
        "status": "planned" if needs_tools else "deliberated"
    }
