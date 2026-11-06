"""Tests for Phase 6: Streaming APIs & Observability Endpoints.

Validates:
1. Real-time SSE streaming generator (metadata, token deltas, completion, error handling).
2. Non-streaming cognitive deliberation endpoint (chat_with_project).
3. Thread management (listing, pagination, message retrieval, deletion).
4. Direct /threads/{id}/messages and /threads/{id}/telemetry endpoints.
5. Out-of-band developer deliberation inspector with zero database bloat.
6. D3.js / vis.js Knowledge Graph visualization with relational fallback.
7. FastAPI route registration under /api/v1.
"""

from __future__ import annotations
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from fastapi import Response
from prisma.enums import MessageRole

from app.main import app
from app.interfaces.api.v1.chat_route import (
    chat_with_project,
    stream_chat_with_project,
    list_project_threads,
    get_thread_messages,
    get_thread_telemetry,
    delete_thread,
    get_thread_messages_direct,
    get_thread_telemetry_direct,
    delete_thread_direct,
    ChatRequest,
    ChatResponseData,
    ThreadSummary,
    _get_or_create_thread,
    _save_user_message,
)
from app.interfaces.api.v1.graph_route import get_project_graph
from app.agents.supervisor.metamo.shared import CognitiveState
from app.agents.supervisor.metamo.eval_logger import RunLogger


# =============================================================================
# 1. FastAPI Route Mounting Tests
# =============================================================================

def test_api_route_mounting():
    """Verifies that all Phase 6 routes are mounted in FastAPI under /api/v1."""
    openapi_paths = set(app.openapi()["paths"].keys())

    # Projects chat & threads routes
    assert "/api/v1/projects/{project_id}/chat" in openapi_paths
    assert "/api/v1/projects/{project_id}/chat/stream" in openapi_paths
    assert "/api/v1/projects/{project_id}/threads" in openapi_paths

    # Standalone threads routes
    assert "/api/v1/threads/{thread_id}/messages" in openapi_paths
    assert "/api/v1/threads/{thread_id}/telemetry" in openapi_paths
    assert "/api/v1/threads/{thread_id}" in openapi_paths

    # Project graph visualizer
    assert "/api/v1/projects/{project_id}/graph" in openapi_paths



# =============================================================================
# 2. Non-Streaming Deliberation Endpoint Tests
# =============================================================================

@pytest.mark.asyncio
async def test_chat_with_project_success():
    """Tests POST /projects/{project_id}/chat orchestrating supervisor graph."""
    mock_prisma = MagicMock()
    mock_project = MagicMock(id="proj-123", userId="user-456")
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)

    mock_thread = MagicMock(
        id="thread-789",
        projectId="proj-123",
        userId="user-456",
        title="Test Chat",
        cognitiveState=json.dumps(CognitiveState().to_dict())
    )
    mock_msg = MagicMock(id="msg-1")

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_response": "Here is the architectural plan.",
        "deliberation_action": "act_respond_direct",
        "style_modifier": "style_thorough",
    })

    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma), \
         patch("app.interfaces.api.v1.chat_route._get_or_create_thread", new=AsyncMock(return_value=mock_thread)), \
         patch("app.interfaces.api.v1.chat_route._save_user_message", new=AsyncMock(return_value=mock_msg)), \
         patch("app.interfaces.api.v1.chat_route.get_compiled_supervisor", new=AsyncMock(return_value=mock_graph)):

        req = ChatRequest(prompt="Explain the microservice architecture.")
        resp = Response()
        result = await chat_with_project(
            project_id="proj-123",
            body=req,
            response=resp,
            request=MagicMock(),
            credentials={"id": "user-456"}
        )

        assert result["success"] is True
        data = result["response"]
        assert isinstance(data, ChatResponseData)
        assert data.answer == "Here is the architectural plan."
        assert data.threadId == "thread-789"
        assert data.cognitiveAction == "act_respond_direct"
        assert data.styleModifier == "style_thorough"


@pytest.mark.asyncio
async def test_chat_with_project_auth_and_not_found():
    """Tests unauthorized, forbidden, and not-found error handling."""
    resp = Response()

    # Unauthorized (no user id)
    res_unauth = await chat_with_project(
        project_id="proj-1",
        body=ChatRequest(prompt="hi"),
        response=resp,
        request=MagicMock(),
        credentials={}
    )
    assert res_unauth["success"] is False
    assert resp.status_code == 401

    # Project not found
    mock_prisma = MagicMock()
    mock_prisma.project.find_unique = AsyncMock(return_value=None)
    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma):
        res_not_found = await chat_with_project(
            project_id="proj-999",
            body=ChatRequest(prompt="hi"),
            response=resp,
            request=MagicMock(),
            credentials={"id": "user-1"}
        )
        assert res_not_found["success"] is False
        assert resp.status_code == 404

    # Forbidden (project belongs to another user)
    mock_other_proj = MagicMock(id="proj-1", userId="other-user")
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_other_proj)
    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma):
        res_forbidden = await chat_with_project(
            project_id="proj-1",
            body=ChatRequest(prompt="hi"),
            response=resp,
            request=MagicMock(),
            credentials={"id": "user-1"}
        )
        assert res_forbidden["success"] is False
        assert resp.status_code == 403


# =============================================================================
# 3. Real-Time SSE Streaming Generator Tests
# =============================================================================

@pytest.mark.asyncio
async def test_stream_chat_with_project_event_lifecycle(tmp_path):
    """Verifies complete SSE stream: metadata -> token chunks -> completion."""
    mock_prisma = MagicMock()
    mock_project = MagicMock(id="proj-1", userId="user-1")
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)

    mock_thread = MagicMock(
        id="thread-stream-1",
        projectId="proj-1",
        userId="user-1",
        title="Stream Thread",
        cognitiveState=json.dumps(CognitiveState().to_dict())
    )
    mock_user_msg = MagicMock(id="msg-u1")
    mock_asst_msg = MagicMock(id="msg-a1")
    mock_prisma.conversationthread.update = AsyncMock()
    mock_prisma.chatmessage.create = AsyncMock(return_value=mock_asst_msg)
    mock_prisma.execute_raw = AsyncMock()

    class MockChunk:
        def __init__(self, text):
            self.text = text

    async def mock_chunk_generator():
        yield MockChunk("Hello ")
        yield MockChunk("from ")
        yield MockChunk("MetaMo!")

    mock_deliberation = {
        "action": "act_clarify",
        "style_modifier": "style_cautious",
        "score_top3": [("act_clarify", 0.85), ("act_respond", 0.40), ("act_verify", 0.35)],
        "reason": "High ambiguity triggered Anti-Premature guard.",
        "signals": {"ambiguity": 0.85, "cx": 0.30},
        "fast_lane": False,
    }

    mock_controller = MagicMock()
    mock_controller.deliberate = AsyncMock(return_value=mock_deliberation)

    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma), \
         patch("app.interfaces.api.v1.chat_route._get_or_create_thread", new=AsyncMock(return_value=mock_thread)), \
         patch("app.interfaces.api.v1.chat_route._save_user_message", new=AsyncMock(return_value=mock_user_msg)), \
         patch("app.interfaces.api.v1.chat_route.retrieve_memories", new=AsyncMock(return_value=[])), \
         patch("app.interfaces.api.v1.chat_route.assemble_context_with_budget", new=AsyncMock(return_value={"tier1_chat_history": [], "kg_context": ""})), \
         patch("app.interfaces.api.v1.chat_route.MetaMoController", return_value=mock_controller), \
         patch("app.interfaces.api.v1.chat_route.genAI.aio.models.generate_content_stream", new=AsyncMock(return_value=mock_chunk_generator())), \
         patch("app.interfaces.api.v1.chat_route.RunLogger") as mock_logger_cls:

        mock_logger_inst = MagicMock()
        mock_logger_cls.return_value = mock_logger_inst

        req = ChatRequest(prompt="Make something cool.")
        streaming_resp = await stream_chat_with_project(
            project_id="proj-1",
            body=req,
            credentials={"id": "user-1"}
        )

        assert streaming_resp.media_type == "text/event-stream"

        events = []
        async for item in streaming_resp.body_iterator:
            events.append(item)

        raw_stream = "".join(events)
        assert "event: metadata" in raw_stream
        assert "event: token" in raw_stream
        assert "event: complete" in raw_stream

        # Verify metadata event content
        assert '"action": "act_clarify"' in raw_stream
        assert '"style": "style_cautious"' in raw_stream
        assert '"thread_id": "thread-stream-1"' in raw_stream

        # Verify token deltas
        assert '{"delta": "Hello "}' in raw_stream
        assert '{"delta": "from "}' in raw_stream
        assert '{"delta": "MetaMo!"}' in raw_stream

        # Verify completion event
        assert '"status": "completed"' in raw_stream
        assert '"message_id": "msg-a1"' in raw_stream

        # Verify out-of-band logger called (zero DB bloat)
        mock_logger_inst.log_turn.assert_called_once()
        log_kwargs = mock_logger_inst.log_turn.call_args[1]
        assert log_kwargs["thread_id"] == "thread-stream-1"
        assert log_kwargs["action"] == "act_clarify"

        # Verify thread state updated and ChatMessage created without bloat
        mock_prisma.conversationthread.update.assert_called_once()
        mock_prisma.chatmessage.create.assert_called_once()
        msg_call_data = mock_prisma.chatmessage.create.call_args[1]["data"]
        assert msg_call_data["content"] == "Hello from MetaMo!"
        assert msg_call_data["cognitiveAction"] == "act_clarify"
        assert msg_call_data["styleModifier"] == "style_cautious"
        assert "cognitiveTelemetry" not in msg_call_data  # Zero DB bloat enforced!


# =============================================================================
# 4. Thread CRUD & Message Pagination Tests
# =============================================================================

@pytest.mark.asyncio
async def test_list_project_threads():
    """Tests GET /projects/{project_id}/threads."""
    now = datetime.now()
    mock_threads = [
        MagicMock(
            id="t1",
            projectId="proj-1",
            title="Sprint Planning",
            createdAt=now,
            updatedAt=now,
            messages=[MagicMock(), MagicMock()]
        ),
        MagicMock(
            id="t2",
            projectId="proj-1",
            title="Database Design",
            createdAt=now,
            updatedAt=now,
            messages=[]
        ),
    ]

    mock_prisma = MagicMock()
    mock_prisma.conversationthread.find_many = AsyncMock(return_value=mock_threads)

    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma):
        resp = Response()
        result = await list_project_threads("proj-1", resp, credentials={"id": "user-1"})

        assert result["success"] is True
        items = result["response"]
        assert len(items) == 2
        assert items[0].id == "t1"
        assert items[0].messageCount == 2
        assert items[1].id == "t2"
        assert items[1].messageCount == 0


@pytest.mark.asyncio
async def test_get_thread_messages_cursor_pagination():
    """Tests GET /threads/{thread_id}/messages with before/after filters."""
    now = datetime.now()
    mock_thread = MagicMock(id="t1", userId="user-1")

    mock_msgs = [
        MagicMock(
            id="m1",
            role=MessageRole.USER,
            content="Hello",
            cognitiveAction=None,
            styleModifier=None,
            createdAt=now
        ),
        MagicMock(
            id="m2",
            role=MessageRole.ASSISTANT,
            content="Hi! How can I help?",
            cognitiveAction="act_respond_direct",
            styleModifier="style_balanced",
            createdAt=now
        ),
    ]

    mock_prisma = MagicMock()
    mock_prisma.conversationthread.find_unique = AsyncMock(return_value=mock_thread)
    mock_prisma.chatmessage.find_many = AsyncMock(return_value=mock_msgs)

    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma):
        resp = Response()
        result = await get_thread_messages_direct("t1", resp, limit=10, credentials={"id": "user-1"})

        assert result["success"] is True
        msgs = result["response"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "USER"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "ASSISTANT"
        assert msgs[1]["cognitiveAction"] == "act_respond_direct"


@pytest.mark.asyncio
async def test_delete_thread():
    """Tests DELETE /threads/{thread_id}."""
    mock_thread = MagicMock(id="t-del", userId="user-1")
    mock_prisma = MagicMock()
    mock_prisma.conversationthread.find_unique = AsyncMock(return_value=mock_thread)
    mock_prisma.conversationthread.delete = AsyncMock()

    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma):
        resp = Response()
        result = await delete_thread_direct("t-del", resp, credentials={"id": "user-1"})

        assert result["success"] is True
        assert result["response"]["deleted"] is True
        assert result["response"]["threadId"] == "t-del"
        mock_prisma.conversationthread.delete.assert_called_once_with(where={"id": "t-del"})


# =============================================================================
# 5. Developer Telemetry Inspector Endpoint Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_thread_telemetry_reads_out_of_band_logs(tmp_path):
    """Tests GET /threads/{thread_id}/telemetry reads file-based JSONL traces."""
    thread_id = "test-telemetry-thread"
    cog_state = {"arousal": 0.45, "turn_count": 2, "over_safety": 0.65}
    mock_thread = MagicMock(
        id=thread_id,
        userId="user-1",
        title="Telemetry Test",
        cognitiveState=json.dumps(cog_state)
    )

    run_dir = tmp_path / "run_20261105_120000"
    run_dir.mkdir(parents=True)
    jsonl_file = run_dir / "turns.jsonl"
    turn_record_1 = {
        "thread_id": thread_id,
        "turn": 1,
        "query": "Init turn",
        "action": "act_clarify",
        "style_modifier": "style_cautious",
        "score_top3": [["act_clarify", 0.90]],
        "reason": "Initial ambiguity"
    }
    turn_record_2 = {
        "thread_id": thread_id,
        "turn": 2,
        "query": "Followup turn",
        "action": "act_respond_direct",
        "style_modifier": "style_balanced",
        "score_top3": [["act_respond_direct", 0.85]],
        "reason": "Clarified goal"
    }
    jsonl_file.write_text(json.dumps(turn_record_1) + "\n" + json.dumps(turn_record_2) + "\n", encoding="utf-8")

    mock_prisma = MagicMock()
    mock_prisma.conversationthread.find_unique = AsyncMock(return_value=mock_thread)

    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.glob", return_value=[run_dir]):

        resp = Response()
        result = await get_thread_telemetry_direct(thread_id, resp, credentials={"id": "user-1"})

        assert result["success"] is True
        data = result["response"]
        assert data["threadId"] == thread_id
        assert data["currentCognitiveState"]["arousal"] == 0.45
        assert len(data["deliberationHistory"]) == 2
        assert data["deliberationHistory"][0]["action"] == "act_clarify"
        assert data["deliberationHistory"][1]["action"] == "act_respond_direct"


# =============================================================================
# 6. D3.js Knowledge Graph Visualizer Endpoint Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_project_graph_neo4j_records():
    """Tests GET /projects/{project_id}/graph with Neo4j records."""
    mock_project = MagicMock(id="p1", userId="u1")
    mock_prisma = MagicMock()
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)

    mock_records = [
        {
            "n": {"id": "n1", "name": "AuthService", "type": "Microservice"},
            "n_labels": ["Component"],
            "r": {"weight": 1.0},
            "rel_type": "DEPENDS_ON",
            "m": {"id": "n2", "name": "UserDatabase", "type": "Postgres"},
            "m_labels": ["Database"],
        }
    ]

    with patch("app.interfaces.api.v1.graph_route.prisma", mock_prisma), \
         patch("app.interfaces.api.v1.graph_route.neo4j_graph_store.execute_query", new=AsyncMock(return_value=mock_records)):

        resp = Response()
        result = await get_project_graph("p1", resp, credentials={"id": "u1"})

        assert result["success"] is True
        graph = result["response"]
        assert graph["projectId"] == "p1"
        assert len(graph["nodes"]) == 2
        assert len(graph["links"]) == 1
        assert graph["links"][0]["source"] == "n1"
        assert graph["links"][0]["target"] == "n2"
        assert graph["links"][0]["type"] == "DEPENDS_ON"
        assert graph["summary"]["nodeCount"] == 2
        assert graph["summary"]["linkCount"] == 1


@pytest.mark.asyncio
async def test_get_project_graph_relational_fallback():
    """Tests GET /projects/{project_id}/graph relational fallback when Neo4j has no nodes."""
    mock_project = MagicMock(id="p1", userId="u1", name="Project Alpha", description="An AI tool", status="ACTIVE")
    mock_req = MagicMock(id="r1", title="User Login", priority="HIGH", status="TODO")
    mock_task = MagicMock(id="t1", title="Create JWT Endpoint", priority="HIGH", status="IN_PROGRESS", requirementId="r1")

    mock_prisma = MagicMock()
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)
    mock_prisma.requirement.find_many = AsyncMock(return_value=[mock_req])
    mock_prisma.task.find_many = AsyncMock(return_value=[mock_task])

    with patch("app.interfaces.api.v1.graph_route.prisma", mock_prisma), \
         patch("app.interfaces.api.v1.graph_route.neo4j_graph_store.execute_query", new=AsyncMock(return_value=[])):

        resp = Response()
        result = await get_project_graph("p1", resp, credentials={"id": "u1"})

        assert result["success"] is True
        graph = result["response"]
        nodes_by_id = {n["id"]: n for n in graph["nodes"]}
        assert "p1" in nodes_by_id
        assert "r1" in nodes_by_id
        assert "t1" in nodes_by_id

        link_types = [link["type"] for link in graph["links"]]
        assert "SPECIFIES" in link_types
        assert "IMPLEMENTS" in link_types
