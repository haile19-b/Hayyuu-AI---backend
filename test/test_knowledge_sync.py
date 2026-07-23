import sys
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.application.use_cases.knowledge_sync_use_cases import knowledge_sync_use_case
from app.infrastructure.vector_store.pgvector import pgvector_store


@pytest.mark.asyncio
async def test_knowledge_synchronization_and_purge():
    doc_id = "sync-doc-999"
    proj_id = "sync-proj-999"
    text = "Requirement A: OAuth2 login support. Requirement B: PostgreSQL storage support."

    class DummyEmbedder:
        def embed_text(self, text: str, model: str = "text-embedding-004"):
            return [0.01] * 768

        def embed_batch(self, texts, model: str = "text-embedding-004"):
            return [[0.01] * 768 for _ in texts]

    class MockGenerateResponse:
        text = '{"users":[], "projects":[], "documents":[], "requirements":[{"id":"r1", "title":"Requirement A", "description":"OAuth2 login support", "type":"FUNCTIONAL", "priority":"P2", "status":"DRAFT"}], "tasks":[{"id":"t1", "title":"Task A", "description":"OAuth2 implementation", "status":"TODO", "priority":"P2"}], "conflicts":[], "relationships":[]}'

    # Ensure PGVector store table is ready
    await pgvector_store.init_vector_store()

    # Mock the LLM generation and Neo4j query methods
    with patch("app.application.use_cases.knowledge_ingestion_use_cases.gemini_embedder", DummyEmbedder()), \
         patch("app.application.use_cases.knowledge_sync_use_cases.neo4j_graph_store.execute_query", new_callable=AsyncMock) as mock_query, \
         patch("app.agents.knowledge_builder.nodes.genAI.models.generate_content") as mock_generate, \
         patch("app.agents.knowledge_builder.graph.write_knowledge_graph_to_neo4j", new_callable=AsyncMock) as mock_write:
        
        mock_generate.return_value = MockGenerateResponse()
        
        # 1. Test update synchronization
        await knowledge_sync_use_case.sync_document_update(
            document_id=doc_id,
            project_id=proj_id,
            raw_text=text,
            filename="specification.pdf"
        )

        # Verify chunks exist in PGVector
        search_res = await pgvector_store.search_similar([0.01] * 768, proj_id)
        assert len(search_res) >= 1
        assert search_res[0]["document_id"] == doc_id

        # Verify Neo4j mock write was called
        assert mock_write.called

        # 2. Test purge hook
        await knowledge_sync_use_case.purge_document_knowledge(doc_id)

        # Verify chunks are removed from PGVector
        search_res_post_purge = await pgvector_store.search_similar([0.01] * 768, proj_id)
        assert len(search_res_post_purge) == 0

        # Verify Neo4j mock query was executed to delete Document
        mock_query.assert_called_with(
            "MATCH (d:Document {id: $document_id}) DETACH DELETE d;",
            {"document_id": doc_id}
        )
