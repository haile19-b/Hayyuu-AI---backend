import sys
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.application.use_cases.knowledge_sync_use_cases import knowledge_sync_use_case


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

    # Mock the LLM generation, embeddings, and Neo4j query methods
    with patch("app.agents.knowledge_builder.nodes.gemini_embedder.embed_batch") as mock_embed, \
         patch("app.infrastructure.graph_store.neo4j.neo4j_graph_store.execute_query", new_callable=AsyncMock) as mock_neo4j, \
         patch("app.infrastructure.graph_store.neo4j.neo4j_graph_store.execute_write_batch", new_callable=AsyncMock) as mock_neo4j_batch, \
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

        # Verify Neo4j mock write was called for chunks (via batch write)
        assert mock_neo4j_batch.called

        # Verify Neo4j mock write was called for project context and link queries
        assert mock_neo4j.called

        # 2. Test purge hook
        await knowledge_sync_use_case.purge_document_knowledge(doc_id)

        # Verify Neo4j mock query was executed to delete Chunks
        mock_neo4j.assert_any_call(
            "MATCH (c:Chunk {document_id: $document_id}) DETACH DELETE c;",
            {"document_id": doc_id}
        )

        # Verify Neo4j mock query was executed to delete Document
        mock_neo4j.assert_any_call(
            "MATCH (d:Document {id: $document_id}) DETACH DELETE d;",
            {"document_id": doc_id}
        )
