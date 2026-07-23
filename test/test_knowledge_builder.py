import sys
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.agents.knowledge_builder.nodes import generate_deterministic_uuid, normalize_title
from app.agents.knowledge_builder.state import KnowledgeBuilderState
from app.agents.knowledge_builder.graph import knowledge_builder_graph


def test_deterministic_id_normalization():
    """Verify entity titles normalize and hash to stable deterministic IDs."""
    title1 = "  User Registration   "
    title2 = "user-registration!!!"
    
    assert normalize_title(title1) == "user registration"
    assert normalize_title(title2) == "user registration"

    id1 = generate_deterministic_uuid("requirement", "proj-123", title1)
    id2 = generate_deterministic_uuid("requirement", "proj-123", title2)
    assert id1 == id2


@pytest.mark.asyncio
async def test_langgraph_agent_execution():
    """Test full LangGraph knowledge builder flow using mock Neo4j & PGVector drivers."""
    raw_text = (
        "Hayyuu AI SRS document context.\n\n"
        "Requirement R1: The system shall support OAuth2 user authentication.\n"
        "Task T1: Implement OAuth2 login handler using python-jose.\n"
        "The OAuth2 login handler Task T1 implements OAuth2 user authentication Requirement R1."
    )

    state = KnowledgeBuilderState(
        project_id="test-proj-001",
        document_id="test-doc-001",
        raw_text=raw_text,
    )

    class MockGenerateResponse:
        text = (
            '{"nodes":['
            '{"id":"r1", "label":"Requirement", "name":"Requirement R1", "description":"support OAuth2 authentication", "properties":[]},'
            '{"id":"t1", "label":"Task", "name":"Task T1", "description":"Implement OAuth2 login", "properties":[]}'
            '],'
            '"relationships":['
            '{"source_id":"t1", "target_id":"r1", "type":"IMPLEMENTS", "properties":[]}'
            ']}'
        )

    # Patch database calls & gemini API calls to avoid network dependency in unit tests
    with patch("app.infrastructure.graph_store.neo4j.neo4j_graph_store.execute_query", new_callable=AsyncMock) as mock_neo4j, \
         patch("app.agents.knowledge_builder.nodes.pgvector_store.upsert_chunks", new_callable=AsyncMock) as mock_vector, \
         patch("app.agents.knowledge_builder.nodes.gemini_embedder.embed_batch") as mock_embed, \
         patch("app.agents.knowledge_builder.nodes.genAI.models.generate_content") as mock_generate:
        
        mock_generate.return_value = MockGenerateResponse()
        mock_embed.return_value = [[0.01] * 768, [0.02] * 768]
        mock_vector.return_value = 2

        # Run graph
        final_state = await knowledge_builder_graph.ainvoke(state)
        
        # Verify extraction worked
        assert "extracted_graph" in final_state
        graph = final_state["extracted_graph"]
        assert len(graph.nodes) >= 2
        assert len(graph.relationships) >= 1

        # Verify execution stats
        assert final_state["total_vectors_stored"] == 2
        assert final_state["total_nodes_stored"] > 0
        assert final_state["status"] == "completed"
        assert len(final_state["errors"]) == 0
