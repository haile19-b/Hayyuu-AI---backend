"""Chat Agent Graph & Streaming Pipeline.

Compiles the Hayyuu Chat Agent StateGraph with PostgreSQL checkpointing,
and provides the async streaming generator `stream_chat_agent` for Server-Sent Events (SSE).
"""

import json
import logging
import asyncio
from typing import Dict, Any, AsyncGenerator, Optional
from langgraph.graph import StateGraph, END
from google.genai import types

from app.core.database import get_checkpointer, get_memory_store, ensure_db_connected
from app.core.ai_cascade import generate_content_stream_cascade
from app.agents.chat_agent.state import ChatAgentState
from app.agents.chat_agent.nodes import (
    assemble_context_node,
    deliberate_node,
    execute_tools_node,
    refine_context_node,
    synthesize_node,
    post_turn_node,
)

logger = logging.getLogger("uvicorn.error")


def route_after_deliberation(state: ChatAgentState) -> str:
    """Conditional Edge: Routes to autonomous tool execution if plan contains pending steps."""
    if state.get("is_plan_complete", True) or not state.get("needs_tools"):
        return "refine_context"
    return "execute_tools"


def route_after_tools(state: ChatAgentState) -> str:
    """Conditional Edge: Loops until all planned investigation needs are satisfied."""
    loop_count = state.get("tool_loop_count", 0)
    is_complete = state.get("is_plan_complete", False)
    if is_complete or loop_count >= 8:
        return "refine_context"
    return "execute_tools"


def build_chat_agent_graph() -> StateGraph:
    """Constructs the Hayyuu Chat Agent StateGraph workflow."""
    workflow = StateGraph(ChatAgentState)

    # 1. Add discrete agent nodes
    workflow.add_node("assemble_context", assemble_context_node)
    workflow.add_node("deliberate", deliberate_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("refine_context", refine_context_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("post_turn", post_turn_node)

    # 2. Configure workflow edges
    workflow.set_entry_point("assemble_context")
    workflow.add_edge("assemble_context", "deliberate")

    # Deliberation -> Execute Tools or Refine Context
    workflow.add_conditional_edges(
        "deliberate",
        route_after_deliberation,
        {
            "execute_tools": "execute_tools",
            "refine_context": "refine_context",
        }
    )

    # Tool Execution Loop -> Next Step or Refine Context
    workflow.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "execute_tools": "execute_tools",
            "refine_context": "refine_context",
        }
    )

    workflow.add_edge("refine_context", "synthesize")
    workflow.add_edge("synthesize", "post_turn")
    workflow.add_edge("post_turn", END)

    return workflow


async def get_compiled_chat_agent(checkpointer=None, store=None):
    """Factory helper returning a compiled chat agent with checkpointing and memory store."""
    builder = build_chat_agent_graph()
    if checkpointer is None:
        try:
            checkpointer = await get_checkpointer()
        except Exception as cp_err:
            logger.debug(f"[ChatAgent] Checkpointer initialization deferred: {cp_err}")
            checkpointer = None

    if store is None:
        try:
            store = await get_memory_store()
        except Exception as store_err:
            logger.debug(f"[ChatAgent] Store initialization deferred: {store_err}")
            store = None

    return builder.compile(checkpointer=checkpointer, store=store)


async def stream_chat_agent(
    project_id: str,
    user_id: str,
    thread_id: str,
    prompt: str
) -> AsyncGenerator[str, None]:
    """
    Executes the Chat Agent workflow while streaming Server-Sent Events (SSE) to the client.
    Emits status, metadata, tool execution, token deltas, and completion events.
    """
    state: ChatAgentState = {
        "messages": [],
        "thread_id": thread_id,
        "project_id": project_id,
        "user_id": user_id,
        "user_query": prompt,
        "user_profile": None,
        "connected_repositories": [],
        "project_documents": [],
        "budgeted_context": {},
        "episodic_memories": [],
        "cognitive_state": {},
        "deliberation_action": "act_respond_direct",
        "deliberation_reason": "",
        "style_modifier": "style_balanced",
        "score_top3": "",
        "signals": {},
        "fast_lane": False,
        "investigation_plan": [],
        "current_plan_step_idx": 0,
        "is_plan_complete": False,
        "structured_evidence": "",
        "pending_tool_calls": [],
        "tool_results": [],
        "tool_loop_count": 0,
        "needs_tools": False,
        "system_instruction": "",
        "gemini_contents": [],
        "final_response": "",
        "created_message_id": None,
        "turn_completed": False,
        "status": "started",
        "error": None,
    }

    try:
        # 1. Context & Memory Retrieval
        yield f"event: status\ndata: {json.dumps({'status': 'retrieving_context', 'message': 'Retrieving project memory and context...'})}\n\n"
        ctx_updates = await assemble_context_node(state)
        state.update(ctx_updates)

        # 2. MetaMo Cognitive Deliberation & Autonomous Plan Formulation
        yield f"event: status\ndata: {json.dumps({'status': 'planning', 'message': 'Analyzing inquiry and structuring investigation plan...'})}\n\n"
        delib_updates = await deliberate_node(state)
        state.update(delib_updates)

        # 3. Emit SSE Metadata Event (Action, Style Modifier, Fast Lane)
        meta_event = {
            "thread_id": thread_id,
            "action": state["deliberation_action"],
            "style": state["style_modifier"],
            "fast_lane": state["fast_lane"],
            "has_plan": bool(state.get("investigation_plan")),
            "plan_steps": len(state.get("investigation_plan", []))
        }
        yield f"event: metadata\ndata: {json.dumps(meta_event)}\n\n"

        # If an investigation plan was formed, notify client
        if state.get("investigation_plan"):
            plan_event = {
                "status": "plan_created",
                "plan": state["investigation_plan"],
                "message": f"Formulated {len(state['investigation_plan'])}-step autonomous investigation plan."
            }
            yield f"event: status\ndata: {json.dumps(plan_event)}\n\n"

        # 4. Plan -> Do Autonomous Execution Loop
        while not state.get("is_plan_complete", True):
            plan = state.get("investigation_plan", [])
            pending = [s for s in plan if s.get("status") == "pending"]
            if not pending or state.get("tool_loop_count", 0) >= 8:
                state["is_plan_complete"] = True
                break

            current_target = pending[0]
            step_event = {
                "status": "executing_step",
                "step_id": current_target.get("id"),
                "description": current_target.get("description"),
                "message": f"Investigating: {current_target.get('description')}..."
            }
            yield f"event: status\ndata: {json.dumps(step_event)}\n\n"

            tool_updates = await execute_tools_node(state)
            state.update(tool_updates)

            for tool_rec in state.get("pending_tool_calls", []):
                t_name = tool_rec.get("tool_name", "tool")
                yield f"event: status\ndata: {json.dumps({'status': 'executing_tool', 'message': f'Executed: {t_name}'})}\n\n"

        # 5. Refine & Rearrange Evidence Node
        if state.get("investigation_plan") or state.get("tool_results"):
            yield f"event: status\ndata: {json.dumps({'status': 'refining_evidence', 'message': 'Structuring and grounding collected findings...'})}\n\n"
        refine_updates = await refine_context_node(state)
        state.update(refine_updates)

        # 6. Final Streaming Synthesis via AI Model Cascade
        yield f"event: status\ndata: {json.dumps({'status': 'generating', 'message': 'Synthesizing response...'})}\n\n"

        full_chunks = []
        gen_config = types.GenerateContentConfig(
            system_instruction=state["system_instruction"],
            temperature=0.3,
        )

        async for chunk, active_model in generate_content_stream_cascade(
            contents=state["gemini_contents"],
            config=gen_config,
            timeout_first_chunk=25.0
        ):
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                full_chunks.append(chunk_text)
                token_event = {"delta": chunk_text}
                yield f"event: token\ndata: {json.dumps(token_event)}\n\n"

        full_response = "".join(full_chunks)
        state["final_response"] = full_response

        # 7. Post-Turn Persistence & Memory Consolidation
        post_updates = await post_turn_node(state)
        state.update(post_updates)

        # 8. Emit SSE Completion Event
        complete_event = {
            "thread_id": thread_id,
            "message_id": state.get("created_message_id"),
            "turn": state.get("cognitive_state", {}).get("turn_count", 1),
            "status": "completed"
        }
        yield f"event: complete\ndata: {json.dumps(complete_event)}\n\n"

    except Exception as e:
        logger.error(f"[ChatAgent: stream_chat_agent] Error during turn: {e}")
        error_event = {"error": str(e)}
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
