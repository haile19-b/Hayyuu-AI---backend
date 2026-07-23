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

    class MockGenerateResponse:
        text = (
            '{"nodes":['
            '{"id":"r1", "label":"Requirement", "name":"Requirement A", "description":"OAuth2 login support", "properties":[]}'
            '],'
            '"relationships":[]}'
        )

    # Ensure PGVector store table is ready
    await pgvector_store.init_vector_store()

    # Mock the LLM generation, embeddings, and Neo4j query methods
    with patch("app.agents.knowledge_builder.nodes.gemini_embedder.embed_batch") as mock_embed, \
         patch("app.infrastructure.graph_store.neo4j.neo4j_graph_store.execute_query", new_callable=AsyncMock) as mock_neo4j, \
         patch("app.agents.knowledge_builder.nodes.genAI.models.generate_content") as mock_generate:
        
        mock_generate.return_value = MockGenerateResponse()
        mock_embed.return_value = [[0.01] * 768]
        
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

        # Verify Neo4j mock write was called for nodes
        assert mock_neo4j.called

        # 2. Test purge hook
        await knowledge_sync_use_case.purge_document_knowledge(doc_id)

        # Verify chunks are removed from PGVector
        search_res_post_purge = await pgvector_store.search_similar([0.01] * 768, proj_id)
        assert len(search_res_post_purge) == 0

        # Verify Neo4j mock query was executed to delete Document
        mock_neo4j.assert_any_call(
            "MATCH (d:Document {id: $document_id}) DETACH DELETE d;",
            {"document_id": doc_id}
        )
