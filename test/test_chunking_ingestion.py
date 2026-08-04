import sys
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.libs.chunker import RecursiveTextSplitter, chunk_document_text, estimate_tokens
from app.application.use_cases.knowledge_ingestion_use_cases import DocumentIngestionUseCase
from app.infrastructure.vector_store.pgvector import pgvector_store


def test_chunker_utility():
    sample_text = (
        "Hayyuu AI Version 1 System Requirements Document.\n\n"
        "1. Functional Requirements:\n"
        "The system shall parse user SRS documents in PDF and DOCX formats. "
        "The system shall store embeddings in PostgreSQL PGVector extension for similarity search.\n\n"
        "2. Non-Functional Requirements:\n"
        "The system API shall respond within 500ms for vector search requests. "
        "The Knowledge Graph shall be maintained using Neo4j bolt driver."
    )

    chunks = chunk_document_text(sample_text, chunk_size_tokens=30, chunk_overlap_tokens=10)
    assert len(chunks) >= 2
    assert chunks[0]["chunk_index"] == 0
    assert "content" in chunks[0]
    assert "token_count" in chunks[0]
    assert chunks[0]["token_count"] > 0


@pytest.mark.asyncio
async def test_document_ingestion_pipeline():
    doc_id = "test-doc-ingest-001"
    proj_id = "test-proj-ingest-001"
    sample_text = (
        "Hayyuu AI Agent Knowledge Builder Ingestion Test.\n\n"
        "The ingestion worker processes raw document text, chunks it using recursive splitting, "
        "and indexes vectors in PostgreSQL using the pgvector extension."
    )

    # Custom mock embedder to avoid requiring real API call in automated test if key is unconfigured
    class DummyEmbedder:
        def embed_text(self, text: str, model: str = "text-embedding-004"):
            return [0.05] * 768

        def embed_batch(self, texts, model: str = "text-embedding-004"):
            return [[0.05] * 768 for _ in texts]

    use_case = DocumentIngestionUseCase()
    use_case.embedder = DummyEmbedder()

    class MockGenerateResponse:
        text = '{"nodes":[], "relationships":[]}'

    with patch("app.agents.knowledge_builder.nodes.gemini_embedder.embed_batch") as mock_embed, \
         patch("app.infrastructure.graph_store.neo4j.neo4j_graph_store.execute_query", new_callable=AsyncMock) as mock_neo4j, \
         patch("app.infrastructure.graph_store.neo4j.neo4j_graph_store.execute_write_batch", new_callable=AsyncMock) as mock_neo4j_batch, \
         patch("app.agents.knowledge_builder.nodes.genAI.models.generate_content") as mock_generate:
        
        mock_generate.return_value = MockGenerateResponse()
        mock_embed.return_value = [[0.05] * 768]
        mock_neo4j_batch.return_value = []

        async def mock_neo4j_dispatch(query, parameters=None):
            q_lower = query.lower()
            if "querynodes" in q_lower:
                return [
                    {
                        "id": "chunk-0",
                        "document_id": doc_id,
                        "project_id": proj_id,
                        "chunk_index": 0,
                        "content": "chunk text",
                        "prev_content": None,
                        "next_content": None,
                        "metadata": '{"filename": "srs.pdf"}',
                        "similarity": 0.95
                    }
                ]
            elif "detach delete" in q_lower and "c:chunk" in q_lower:
                return [{"cnt": 1}]
            return []

        mock_neo4j.side_effect = mock_neo4j_dispatch

        # Perform ingestion
        res = await use_case.ingest_document_text(
            document_id=doc_id,
            project_id=proj_id,
            raw_text=sample_text,
            extra_metadata={"filename": "srs.pdf"},
            chunk_size_tokens=50,
            chunk_overlap_tokens=10,
        )

        assert res["status"] == "indexed"
        assert res["total_chunks"] >= 1
        assert mock_neo4j.called

        # Search knowledge
        search_results = await use_case.search_document_knowledge(
            query_text="knowledge builder",
            project_id=proj_id,
            top_k=2,
        )
        assert len(search_results) >= 1
        assert search_results[0]["document_id"] == doc_id

        # Remove knowledge
        deleted = await use_case.remove_document_knowledge(doc_id)
        assert deleted >= 1


def test_live_gemini_embedding_api():
    """Verify live Gemini gemini-embedding-2 api works if key is present."""
    from app.core.env import settings
    from app.infrastructure.ai.gemini_embedder import gemini_embedder

    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not set. Skipping live API test.")

    single_emb = gemini_embedder.embed_text("Verify this search query")
    assert len(single_emb) == 768
    assert isinstance(single_emb[0], float)

    batch_embs = gemini_embedder.embed_batch(["First test document chunk", "Second test document chunk"])
    assert len(batch_embs) == 2
    assert len(batch_embs[0]) == 768
    assert len(batch_embs[1]) == 768

