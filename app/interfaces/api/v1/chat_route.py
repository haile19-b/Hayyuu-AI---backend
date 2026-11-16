"""Streaming SSE & Multi-Turn Cognitive Chat Endpoints.

Integrates MetaMo cognitive deliberation, LangGraph checkpointer persistence,
multi-tier context budgeting, and out-of-band evaluation telemetry.
"""

from __future__ import annotations
import json
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, Response, status, Request
from fastapi.responses import StreamingResponse

from google.genai import types

from app.core.database import prisma, ensure_db_connected
from app.middleware.auth_middleware import auth_middleware
from app.domain.entities.auth.auth_schema import ApiResponse
from prisma.enums import MessageRole
from app.agents.chat_agent import stream_chat_agent

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
    await ensure_db_connected()
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
    await ensure_db_connected()
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
        await ensure_db_connected()
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

        user_profile = None
        try:
            user_rec = await prisma.user.find_unique(where={"id": user_id})
            if user_rec:
                user_profile = {
                    "fullName": user_rec.fullName,
                    "userName": user_rec.userName,
                    "email": user_rec.email,
                }
        except Exception:
            pass

        initial_state: SupervisorState = {
            "messages": [],
            "thread_id": thread.id,
            "project_id": project_id,
            "user_id": user_id,
            "user_query": body.prompt,
            "user_profile": user_profile,
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

        # Trigger autonomous background episodic memory consolidation
        try:
            from app.core.queue import enqueue_memory_consolidation
            await enqueue_memory_consolidation(user_id=user_id, project_id=project_id, thread_id=thread.id)
        except Exception as q_err:
            logger.debug(f"Memory consolidation trigger deferred: {q_err}")

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
    Delegates context assembly, MetaMo deliberation, dynamic MCP tool execution,
    model cascade streaming synthesis, and persistence to Hayyuu Chat Agent.
    """
    await ensure_db_connected()
    user_id = credentials.get("id") if credentials else None
    if not user_id:
        return Response("Unauthorized", status_code=status.HTTP_401_UNAUTHORIZED)

    project = await prisma.project.find_unique(where={"id": project_id})
    if not project or project.userId != user_id:
        return Response("Unauthorized", status_code=status.HTTP_403_FORBIDDEN)

    thread = await _get_or_create_thread(project_id, user_id, body.threadId, body.prompt)
    await _save_user_message(thread.id, body.prompt)

    return StreamingResponse(
        stream_chat_agent(
            project_id=project_id,
            user_id=user_id,
            thread_id=thread.id,
            prompt=body.prompt
        ),
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


