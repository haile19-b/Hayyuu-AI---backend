"""Streaming SSE & Multi-Turn Cognitive Chat Endpoints.

Integrates MetaMo cognitive deliberation, LangGraph checkpointer persistence,
multi-tier context budgeting, and out-of-band evaluation telemetry.
"""

from __future__ import annotations
import json
import logging
import time
from typing import Optional, List, Dict, Any, AsyncGenerator
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, Response, status, Request
from fastapi.responses import StreamingResponse

from google.genai import types

from app.core.config import genAI
from app.core.env import settings
from app.core.database import prisma, get_checkpointer, get_memory_store
from app.middleware.auth_middleware import auth_middleware
from app.domain.entities.auth.auth_schema import ApiResponse
from prisma.enums import MessageRole

from app.agents.supervisor.state import SupervisorState
from app.agents.supervisor.orchestrator import (
    build_supervisor_graph,
    get_compiled_supervisor,
    STYLE_PROMPTS,
    SYSTEM_BASE_PROMPT
)
from app.agents.supervisor.context_budget import assemble_context_with_budget
from app.agents.supervisor.metamo.shared import CognitiveState
from app.agents.supervisor.metamo.controller import MetaMoController
from app.agents.supervisor.metamo.homeostasis import post_turn_update
from app.agents.supervisor.metamo.eval_logger import RunLogger
from app.agents.memory.retriever import retrieve_memories, format_memories_for_prompt

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/projects", tags=["Chat"])


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The user's query or message.")
    threadId: Optional[str] = Field(None, description="Optional conversation thread UUID.")
    documentId: Optional[str] = Field(None, description="Optional document UUID filter.")
    limit: Optional[int] = Field(5, description="Search result limit.")


class ChatResponseData(BaseModel):
    answer: str
    threadId: str
    cognitiveAction: Optional[str] = None
    styleModifier: Optional[str] = None
    sources: List[Any] = []
    nodes: List[Any] = []


class ThreadSummary(BaseModel):
    id: str
    projectId: str
    title: Optional[str]
    createdAt: str
    updatedAt: str
    messageCount: int = 0


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

async def _get_or_create_thread(project_id: str, user_id: str, thread_id: Optional[str], initial_prompt: str):
    """Retrieves an existing ConversationThread or creates a new initialized one."""
    if thread_id:
        existing = await prisma.conversationthread.find_unique(where={"id": thread_id})
        if existing and existing.projectId == project_id and existing.userId == user_id:
            return existing

    # Create new thread
    title = (initial_prompt[:50] + "...") if len(initial_prompt) > 50 else initial_prompt
    initial_cog = CognitiveState().to_dict()
    new_thread = await prisma.conversationthread.create(
        data={
            "projectId": project_id,
            "userId": user_id,
            "title": title,
            "cognitiveState": json.dumps(initial_cog)
        }
    )
    return new_thread


async def _save_user_message(thread_id: str, content: str):
    """Creates a user message and queues vector embedding creation."""
    msg = await prisma.chatmessage.create(
        data={
            "threadId": thread_id,
            "role": MessageRole.USER,
            "content": content
        }
    )
    try:
        from app.infrastructure.ai.gemini_embedder import GeminiEmbedder
        embedder = GeminiEmbedder()
        vec = embedder.embed_text(content, user_query=True)
        vec_lit = "[" + ",".join(str(x) for x in vec) + "]"
        await prisma.execute_raw(
            f'UPDATE chat_messages SET embedding = \'{vec_lit}\'::vector WHERE id = $1;',
            msg.id
        )
    except Exception as e:
        logger.debug(f"User message vector embedding deferred: {e}")
    return msg


# -----------------------------------------------------------------------------
# 1. Non-Streaming Cognitive Deliberation Endpoint
# -----------------------------------------------------------------------------

@router.post("/{project_id}/chat", response_model=ApiResponse[ChatResponseData])
async def chat_with_project(
    project_id: str,
    body: ChatRequest,
    response: Response,
    request: Request,
    credentials = Depends(auth_middleware)
):
    """
    Standard HTTP endpoint orchestrating the MetaMo Cognitive Supervisor.
    Persists checkpoints to LangGraph AsyncPostgresSaver and episodic memory to Store.
    """
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        project = await prisma.project.find_unique(where={"id": project_id})
        if not project:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"success": False, "error": "Project not found"}
        if project.userId != user_id:
            response.status_code = status.HTTP_403_FORBIDDEN
            return {"success": False, "error": "Unauthorized"}

        thread = await _get_or_create_thread(project_id, user_id, body.threadId, body.prompt)
        await _save_user_message(thread.id, body.prompt)

        # Parse cognitive state
        cog_state_dict = None
        if thread.cognitiveState:
            try:
                cog_state_dict = json.loads(thread.cognitiveState) if isinstance(thread.cognitiveState, str) else thread.cognitiveState
            except Exception:
                pass

        # Compile and execute supervisor graph
        supervisor_graph = await get_compiled_supervisor()
        config = {"configurable": {"thread_id": thread.id}}

        initial_state: SupervisorState = {
            "messages": [],
            "thread_id": thread.id,
            "project_id": project_id,
            "user_id": user_id,
            "user_query": body.prompt,
            "cognitive_state": cog_state_dict or CognitiveState().to_dict(),
            "current_signals": {},
            "deliberation_action": "",
            "score_top3": "",
            "deliberation_reason": "",
            "style_modifier": "style_balanced",
            "fast_lane": False,
            "budgeted_context": {},
            "episodic_memories": [],
            "subagent_output": None,
            "tool_calls_executed": [],
            "final_response": "",
            "turn_count": 0,
            "review_required": False,
            "review_status": None
        }

        result = await supervisor_graph.ainvoke(initial_state, config=config)

        resp_data = ChatResponseData(
            answer=result.get("final_response", ""),
            threadId=thread.id,
            cognitiveAction=result.get("deliberation_action"),
            styleModifier=result.get("style_modifier"),
            sources=[],
            nodes=[]
        )

        return {"success": True, "response": resp_data}

    except Exception as e:
        logger.error(f"Error in chat_with_project: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


# -----------------------------------------------------------------------------
# 2. Real-Time Streaming SSE Endpoint
# -----------------------------------------------------------------------------

@router.post("/{project_id}/chat/stream")
async def stream_chat_with_project(
    project_id: str,
    body: ChatRequest,
    credentials = Depends(auth_middleware)
):
    """
    High-performance Server-Sent Events (SSE) streaming endpoint.
    Emits metadata, token deltas, and completion events.
    Deliberation telemetry is logged out-of-band to CSV/JSONL for developers.
    """
    user_id = credentials.get("id") if credentials else None
    if not user_id:
        return Response("Unauthorized", status_code=status.HTTP_401_UNAUTHORIZED)

    project = await prisma.project.find_unique(where={"id": project_id})
    if not project or project.userId != user_id:
        return Response("Unauthorized", status_code=status.HTTP_403_FORBIDDEN)

    thread = await _get_or_create_thread(project_id, user_id, body.threadId, body.prompt)
    await _save_user_message(thread.id, body.prompt)

    async def event_generator() -> AsyncGenerator[str, None]:
        t0 = time.time()
        full_content_chunks: List[str] = []

        try:
            # 1. Multi-tier Context & Memory Retrieval
            memories = await retrieve_memories(user_id=user_id, project_id=project_id, query=body.prompt, limit=5)
            budgeted = await assemble_context_with_budget(
                thread_id=thread.id,
                project_id=project_id,
                user_query=body.prompt,
                tier1_tokens=2000,
                tier2_tokens=1500,
                kg_tokens=1500,
                semantic_search=True,
                memories=memories,
            )

            # Load cognitive state
            cog_state_dict = None
            if thread.cognitiveState:
                try:
                    cog_state_dict = json.loads(thread.cognitiveState) if isinstance(thread.cognitiveState, str) else thread.cognitiveState
                except Exception:
                    pass
            cog_state = cog_state_dict or CognitiveState().to_dict()

            # 2. MetaMo Cognitive Deliberation
            controller = MetaMoController()
            deliberation = await controller.deliberate(
                user_query=body.prompt,
                chat_history=budgeted.get("tier1_chat_history", []),
                kg_context=budgeted.get("kg_context", ""),
                state=cog_state
            )

            latency_ms = (time.time() - t0) * 1000.0

            # 3. Log Evaluation Telemetry Out-of-Band (Zero DB Bloat)
            try:
                run_logger = RunLogger()
                run_logger.log_turn(
                    thread_id=thread.id,
                    turn=int(cog_state.get("turn_count", 0)) + 1,
                    query=body.prompt,
                    action=deliberation["action"],
                    style_modifier=deliberation["style_modifier"],
                    score_top3=deliberation["score_top3"],
                    reason=deliberation["reason"],
                    cognitive_state=cog_state,
                    latency_ms=latency_ms
                )
            except Exception as log_err:
                logger.debug(f"Telemetry logging bypassed: {log_err}")

            # 4. Emit SSE Metadata Event (Action & Style only, no internal telemetry)
            meta_event = {
                "thread_id": thread.id,
                "action": deliberation["action"],
                "style": deliberation["style_modifier"],
                "fast_lane": deliberation["fast_lane"]
            }
            yield f"event: metadata\ndata: {json.dumps(meta_event)}\n\n"

            # 5. Build Dynamic System Prompt with Action Directive, Style Modifier & Memory
            style_key = deliberation["style_modifier"]
            style_directive = STYLE_PROMPTS.get(style_key, STYLE_PROMPTS["style_balanced"])
            kg_text = budgeted.get("kg_context", "")

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

            mem_formatted = format_memories_for_prompt(memories) if memories else ""
            episodic_block = f"Long-Term User Preferences & Decisions:\n{mem_formatted}" if mem_formatted else ""

            system_instruction = SYSTEM_BASE_PROMPT.format(
                style_directive=style_directive,
                domain_context=f"Domain Context:\n{kg_text}" if kg_text else "",
                episodic_memory_context=episodic_block
            ) + action_directive

            # 6. Stream LLM Response via Gemini Async Streaming API
            model_name = settings.GEMINI_MODELS[0] if settings.GEMINI_MODELS else "gemini-3.8-flash"
            try:
                response_stream = await genAI.aio.models.generate_content_stream(
                    model=model_name,
                    contents=[body.prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    )
                )

                async for chunk in response_stream:
                    if chunk.text:
                        full_content_chunks.append(chunk.text)
                        token_event = {"delta": chunk.text}
                        yield f"event: token\ndata: {json.dumps(token_event)}\n\n"
            except Exception as stream_err:
                logger.warning(f"Async streaming fallback: {stream_err}")
                fallback_msg = f"Response generated under {deliberation['style_modifier']}."
                full_content_chunks.append(fallback_msg)
                yield f"event: token\ndata: {json.dumps({'delta': fallback_msg})}\n\n"

            full_response = "".join(full_content_chunks)

            # 7. Post-Turn Drive & Homeostasis Update
            updated_cog_state = post_turn_update(
                state=cog_state,
                signals=deliberation["signals"],
                decision={"action": deliberation["action"], "style_modifier": deliberation["style_modifier"]},
                turn_success=True
            )

            # Persist thread state and assistant message (zero DB bloat: no telemetry in ChatMessage)
            await prisma.conversationthread.update(
                where={"id": thread.id},
                data={"cognitiveState": json.dumps(updated_cog_state)}
            )

            created_msg = await prisma.chatmessage.create(
                data={
                    "threadId": thread.id,
                    "role": MessageRole.ASSISTANT,
                    "content": full_response,
                    "cognitiveAction": deliberation["action"],
                    "styleModifier": deliberation["style_modifier"],
                }
            )

            # Generate embedding in background
            try:
                from app.infrastructure.ai.gemini_embedder import GeminiEmbedder
                embedder = GeminiEmbedder()
                vec = embedder.embed_text(full_response, user_query=False)
                vec_lit = "[" + ",".join(str(x) for x in vec) + "]"
                await prisma.execute_raw(
                    f'UPDATE chat_messages SET embedding = \'{vec_lit}\'::vector WHERE id = $1;',
                    created_msg.id
                )
            except Exception:
                pass

            # 8. Emit SSE Completion Event
            complete_event = {
                "thread_id": thread.id,
                "message_id": created_msg.id,
                "turn": updated_cog_state.get("turn_count", 1),
                "status": "completed"
            }
            yield f"event: complete\ndata: {json.dumps(complete_event)}\n\n"

        except Exception as e:
            logger.error(f"Error in SSE stream generation: {e}")
            error_event = {"error": str(e)}
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# -----------------------------------------------------------------------------
# 3. Thread Management Endpoints
# -----------------------------------------------------------------------------

@router.get("/{project_id}/threads", response_model=ApiResponse[List[ThreadSummary]])
async def list_project_threads(
    project_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """Lists all conversation threads associated with this project and authenticated user."""
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        threads = await prisma.conversationthread.find_many(
            where={"projectId": project_id, "userId": user_id},
            order={"updatedAt": "desc"},
            include={"messages": True}
        )

        summaries = [
            ThreadSummary(
                id=t.id,
                projectId=t.projectId,
                title=t.title,
                createdAt=t.createdAt.isoformat(),
                updatedAt=t.updatedAt.isoformat(),
                messageCount=len(t.messages) if t.messages else 0
            )
            for t in threads
        ]

        return {"success": True, "response": summaries}
    except Exception as e:
        logger.error(f"Error listing threads: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    response: Response,
    limit: int = 50,
    before: Optional[str] = None,
    after: Optional[str] = None,
    credentials = Depends(auth_middleware)
):
    """Retrieves chronological chat turns with optional cursor pagination."""
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        thread = await prisma.conversationthread.find_unique(where={"id": thread_id})
        if not thread or thread.userId != user_id:
            response.status_code = status.HTTP_403_FORBIDDEN
            return {"success": False, "error": "Unauthorized"}

        where: Dict[str, Any] = {"threadId": thread_id}
        if before:
            ref = await prisma.chatmessage.find_unique(where={"id": before})
            if ref:
                where["createdAt"] = {"lt": ref.createdAt}
        elif after:
            ref = await prisma.chatmessage.find_unique(where={"id": after})
            if ref:
                where["createdAt"] = {"gt": ref.createdAt}

        messages = await prisma.chatmessage.find_many(
            where=where,
            order={"createdAt": "asc"},
            take=min(limit, 100)
        )

        formatted = [
            {
                "id": m.id,
                "role": m.role.value if hasattr(m.role, "value") else str(m.role).upper(),
                "content": m.content,
                "cognitiveAction": m.cognitiveAction,
                "styleModifier": m.styleModifier,
                "createdAt": m.createdAt.isoformat()
            }
            for m in messages
        ]

        return {"success": True, "response": formatted}
    except Exception as e:
        logger.error(f"Error fetching thread messages: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.get("/threads/{thread_id}/telemetry")
async def get_thread_telemetry(
    thread_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """
    Developer observability endpoint.
    Retrieves out-of-band evaluation trajectories and cognitive states for a thread
    directly from evaluation logs and thread state.
    """
    try:
        from pathlib import Path
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        thread = await prisma.conversationthread.find_unique(where={"id": thread_id})
        if not thread or thread.userId != user_id:
            response.status_code = status.HTTP_403_FORBIDDEN
            return {"success": False, "error": "Unauthorized"}

        current_cog = {}
        if thread.cognitiveState:
            try:
                current_cog = json.loads(thread.cognitiveState) if isinstance(thread.cognitiveState, str) else thread.cognitiveState
            except Exception:
                pass

        # Read deliberation traces from logs/eval/
        base_logs = Path(__file__).resolve().parent.parent.parent.parent.parent / "logs" / "eval"
        turns: List[Dict[str, Any]] = []

        if base_logs.exists():
            for run_dir in sorted(base_logs.glob("run_*"), reverse=True):
                jsonl_file = run_dir / "turns.jsonl"
                if jsonl_file.exists():
                    try:
                        with jsonl_file.open("r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    rec = json.loads(line)
                                    if rec.get("thread_id") == thread_id:
                                        turns.append(rec)
                    except Exception:
                        pass

        # Sort chronological
        turns.sort(key=lambda t: t.get("turn", 0))

        return {
            "success": True,
            "response": {
                "threadId": thread_id,
                "title": thread.title,
                "currentCognitiveState": current_cog,
                "deliberationHistory": turns,
                "turnCount": len(turns) or current_cog.get("turn_count", 0)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching thread telemetry: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """Deletes a conversation thread and all cascaded chat messages."""
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        thread = await prisma.conversationthread.find_unique(where={"id": thread_id})
        if not thread or thread.userId != user_id:
            response.status_code = status.HTTP_403_FORBIDDEN
            return {"success": False, "error": "Unauthorized"}

        await prisma.conversationthread.delete(where={"id": thread_id})
        return {"success": True, "response": {"deleted": True, "threadId": thread_id}}
    except Exception as e:
        logger.error(f"Error deleting thread: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


# -----------------------------------------------------------------------------
# 4. Standalone Thread Router (/threads)
# -----------------------------------------------------------------------------

thread_router = APIRouter(prefix="/threads", tags=["Threads"])


@thread_router.get("/{thread_id}/messages")
async def get_thread_messages_direct(
    thread_id: str,
    response: Response,
    limit: int = 50,
    before: Optional[str] = None,
    after: Optional[str] = None,
    credentials = Depends(auth_middleware)
):
    """Retrieves chronological chat turns via /threads/{id}/messages."""
    return await get_thread_messages(
        thread_id=thread_id,
        response=response,
        limit=limit,
        before=before,
        after=after,
        credentials=credentials
    )


@thread_router.get("/{thread_id}/telemetry")
async def get_thread_telemetry_direct(
    thread_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """Retrieves out-of-band evaluation telemetry via /threads/{id}/telemetry."""
    return await get_thread_telemetry(
        thread_id=thread_id,
        response=response,
        credentials=credentials
    )


@thread_router.delete("/{thread_id}")
async def delete_thread_direct(
    thread_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """Deletes a conversation thread via /threads/{id}."""
    return await delete_thread(
        thread_id=thread_id,
        response=response,
        credentials=credentials
    )


