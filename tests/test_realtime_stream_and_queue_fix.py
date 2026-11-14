"""Tests verifying the real-time SSE typed status events, streaming fallbacks,
and ARQ queue deduplication / deletion cleanup fixes.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prisma.enums import DocumentStatus, MessageRole
from app.interfaces.api.v1.chat_route import stream_chat_with_project, ChatRequest
from app.core.queue import enqueue_document_analysis, analyze_document_job, abort_document_analysis
from app.domain.entities.documents.documents_service import DocumentsService
from app.agents.supervisor.metamo.shared import CognitiveState


@pytest.mark.asyncio
async def test_stream_chat_emits_typed_status_events():
    """Verifies that the streaming endpoint emits distinct typed status events:
    retrieving_context -> deliberating -> metadata -> generating -> token -> complete.
    """
    mock_prisma = MagicMock()
    mock_project = MagicMock(id="p1", userId="u1")
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)

    mock_thread = MagicMock(
        id="t1",
        projectId="p1",
        userId="u1",
        title="Test Thread",
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

    async def mock_chunk_stream():
        yield MockChunk("Real-time ")
        yield MockChunk("stream response.")

    mock_deliberation = {
        "action": "act_respond_direct",
        "style_modifier": "style_balanced",
        "score_top3": [("act_respond_direct", 0.9)],
        "reason": "Direct query",
        "signals": {"ambiguity": 0.1},
        "fast_lane": True,
    }

    mock_controller = MagicMock()
    mock_controller.deliberate = AsyncMock(return_value=mock_deliberation)

    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma), \
         patch("app.interfaces.api.v1.chat_route.ensure_db_connected", new=AsyncMock()), \
         patch("app.interfaces.api.v1.chat_route._get_or_create_thread", new=AsyncMock(return_value=mock_thread)), \
         patch("app.interfaces.api.v1.chat_route._save_user_message", new=AsyncMock(return_value=mock_user_msg)), \
         patch("app.interfaces.api.v1.chat_route.retrieve_memories", new=AsyncMock(return_value=[])), \
         patch("app.interfaces.api.v1.chat_route.assemble_context_with_budget", new=AsyncMock(return_value={"tier1_chat_history": [], "kg_context": ""})), \
         patch("app.interfaces.api.v1.chat_route.MetaMoController", return_value=mock_controller), \
         patch("app.interfaces.api.v1.chat_route.genAI.aio.models.generate_content_stream", new=AsyncMock(return_value=mock_chunk_stream())), \
         patch("app.infrastructure.ai.gemini_embedder.GeminiEmbedder.embed_text", return_value=[0.1] * 768), \
         patch("app.interfaces.api.v1.chat_route.RunLogger"):

        req = ChatRequest(prompt="What backend architecture do we use?")
        streaming_resp = await stream_chat_with_project(
            project_id="p1",
            body=req,
            credentials={"id": "u1"}
        )

        events = []
        async for chunk in streaming_resp.body_iterator:
            events.append(chunk)

        raw = "".join(events)

        # 1. Verify status events
        assert "event: status" in raw
        assert '"status": "retrieving_context"' in raw
        assert '"status": "deliberating"' in raw
        assert '"status": "generating"' in raw

        # 2. Verify metadata event
        assert "event: metadata" in raw
        assert '"action": "act_respond_direct"' in raw

        # 3. Verify token events
        assert "event: token" in raw
        assert '{"delta": "Real-time "}' in raw
        assert '{"delta": "stream response."}' in raw

        # 4. Verify complete event
        assert "event: complete" in raw
        assert '"status": "completed"' in raw
        assert '"message_id": "msg-a1"' in raw

        # Verify NO dummy fallback string was emitted
        assert "Response generated under" not in raw


@pytest.mark.asyncio
async def test_stream_chat_model_fallback_cascade():
    """Verifies that if the primary model fails, the stream cascades to backup models."""
    mock_prisma = MagicMock()
    mock_project = MagicMock(id="p1", userId="u1")
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)

    mock_thread = MagicMock(
        id="t1",
        projectId="p1",
        userId="u1",
        title="Test Thread",
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

    async def mock_backup_stream():
        yield MockChunk("Fallback model content.")

    mock_deliberation = {
        "action": "act_respond_direct",
        "style_modifier": "style_balanced",
        "score_top3": [],
        "reason": "",
        "signals": {},
        "fast_lane": True,
    }
    mock_controller = MagicMock()
    mock_controller.deliberate = AsyncMock(return_value=mock_deliberation)

    call_count = 0
    async def mock_generate_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Primary model gemini-3.8-flash 503 Overloaded")
        return mock_backup_stream()

    with patch("app.interfaces.api.v1.chat_route.prisma", mock_prisma), \
         patch("app.interfaces.api.v1.chat_route.ensure_db_connected", new=AsyncMock()), \
         patch("app.interfaces.api.v1.chat_route._get_or_create_thread", new=AsyncMock(return_value=mock_thread)), \
         patch("app.interfaces.api.v1.chat_route._save_user_message", new=AsyncMock(return_value=mock_user_msg)), \
         patch("app.interfaces.api.v1.chat_route.retrieve_memories", new=AsyncMock(return_value=[])), \
         patch("app.interfaces.api.v1.chat_route.assemble_context_with_budget", new=AsyncMock(return_value={"tier1_chat_history": [], "kg_context": ""})), \
         patch("app.interfaces.api.v1.chat_route.MetaMoController", return_value=mock_controller), \
         patch("app.interfaces.api.v1.chat_route.genAI.aio.models.generate_content_stream", side_effect=mock_generate_stream), \
         patch("app.infrastructure.ai.gemini_embedder.GeminiEmbedder.embed_text", return_value=[0.1] * 768), \
         patch("app.interfaces.api.v1.chat_route.RunLogger"):

        req = ChatRequest(prompt="Test fallback")
        streaming_resp = await stream_chat_with_project(
            project_id="p1",
            body=req,
            credentials={"id": "u1"}
        )

        events = []
        async for chunk in streaming_resp.body_iterator:
            events.append(chunk)

        raw = "".join(events)
        assert '"status": "retrying"' in raw
        assert '{"delta": "Fallback model content."}' in raw
        assert '"status": "completed"' in raw
        assert call_count >= 2


@pytest.mark.asyncio
async def test_enqueue_document_analysis_job_id_deduplication():
    """Verifies that enqueue_document_analysis passes a deterministic _job_id to ARQ."""
    from arq.jobs import JobStatus
    mock_pool = MagicMock()
    mock_pool.delete = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="doc_analysis_doc-123"))

    with patch("app.core.queue.redis_pool", mock_pool), \
         patch("arq.jobs.Job.status", new=AsyncMock(return_value=JobStatus.not_found)):
        res = await enqueue_document_analysis("doc-123", "proj-456")
        assert res is True
        mock_pool.enqueue_job.assert_called_once_with(
            "analyze_document_job",
            "doc-123",
            "proj-456",
            _job_id="doc_analysis_doc-123"
        )


@pytest.mark.asyncio
async def test_enqueue_document_analysis_force_retry_clears_keys():
    """Verifies that force=True on retry flushes Redis keys and enqueues fresh job."""
    from arq.jobs import JobStatus
    mock_pool = MagicMock()
    mock_pool.delete = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="doc_analysis_doc-123"))

    with patch("app.core.queue.redis_pool", mock_pool), \
         patch("arq.jobs.Job.status", new=AsyncMock(return_value=JobStatus.complete)):
        res = await enqueue_document_analysis("doc-123", "proj-456", force=True)
        assert res is True
        mock_pool.delete.assert_called_once()
        mock_pool.enqueue_job.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_document_job_deleted_doc_clean_discard():
    """Verifies that if a document was deleted before or during analysis,
    the job exits cleanly without raising an exception (preventing ARQ retry loops).
    """
    mock_prisma = MagicMock()
    mock_prisma.document.find_unique = AsyncMock(return_value=None)

    with patch("app.core.database.prisma", mock_prisma), \
         patch("app.core.database.ensure_db_connected", new=AsyncMock()), \
         patch("app.agents.requirements_extractor.graph.run_workflow", new=AsyncMock()) as mock_run:

        # Should return cleanly without throwing
        await analyze_document_job(None, "deleted-doc-id", "proj-456")
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_delete_document_aborts_arq_job_and_cleans_checkpoints():
    """Verifies that deleting a document aborts the ARQ job and cleans checkpointer records."""
    mock_prisma = MagicMock()
    mock_doc = MagicMock(id="doc-999", projectId="proj-1", filePath="projects/p1/doc.pdf")
    mock_prisma.document.find_unique = AsyncMock(return_value=mock_doc)
    mock_prisma.document.delete = AsyncMock()
    mock_prisma.execute_raw = AsyncMock()

    with patch("app.domain.entities.documents.documents_service.ProjectsService.get_project_by_id", new=AsyncMock(return_value={"success": True})), \
         patch("app.domain.entities.documents.documents_service.prisma", mock_prisma), \
         patch("app.domain.entities.documents.documents_service.storage_utility.delete_file", new=AsyncMock()), \
         patch("app.domain.entities.documents.documents_service.abort_document_analysis", new=AsyncMock()) as mock_abort:

        res = await DocumentsService.delete_document("proj-1", "doc-999", "user-1")
        assert res["success"] is True

        # Verify abort was called with document_id
        mock_abort.assert_called_once_with("doc-999")

        # Verify checkpointer records were cleaned
        assert mock_prisma.execute_raw.call_count >= 3
        mock_prisma.document.delete.assert_called_once_with(where={"id": "doc-999"})


@pytest.mark.asyncio
async def test_knowledge_builder_skips_when_already_indexed():
    """Verifies that run_knowledge_builder skips re-running extraction when Neo4j already has chunks."""
    from app.agents.knowledge_builder.graph import run_knowledge_builder
    from app.agents.knowledge_builder.state import KnowledgeBuilderState

    mock_prisma = MagicMock()
    mock_prisma.aisuggestion.find_first = AsyncMock(return_value=None)

    mock_neo4j = MagicMock()
    mock_neo4j.execute_query = AsyncMock(return_value=[{"cnt": 12}])

    state = KnowledgeBuilderState(
        project_id="p1",
        document_id="doc-123",
        filename="test.pdf"
    )

    with patch("app.agents.knowledge_builder.graph.prisma", mock_prisma), \
         patch("app.agents.knowledge_builder.graph.ensure_db_connected", new=AsyncMock()), \
         patch("app.infrastructure.graph_store.neo4j.neo4j_graph_store.execute_query", mock_neo4j.execute_query), \
         patch("app.agents.knowledge_builder.graph.workflow.compile") as mock_compile:

        await run_knowledge_builder(state)

        # Should return early without compiling or invoking the graph
        mock_compile.assert_not_called()

