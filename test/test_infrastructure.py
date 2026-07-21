import sys
import asyncio
import pytest
import pytest_asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.infrastructure.vector_store.pgvector import PGVectorStore, pgvector_store
from app.infrastructure.graph_store.neo4j import Neo4jGraphStore, neo4j_graph_store


@pytest.mark.asyncio
async def test_pgvector_store_operations():
    """Test PGVector initialization, chunk upsert, similarity search, and deletion."""
    # 1. Initialize DB extension and table
    await pgvector_store.init_vector_store()

    doc_id = "test-doc-001"
    proj_id = "test-proj-001"
    dummy_vector = [0.1] * 768

    chunks = [
        {
            "chunk_index": 0,
            "content": "This is a test document chunk regarding user requirements.",
            "embedding": dummy_vector,
            "metadata": {"source": "test_spec.pdf", "page": 1},
        },
        {
            "chunk_index": 1,
            "content": "This is the second chunk about system database architecture.",
            "embedding": [0.2] * 768,
            "metadata": {"source": "test_spec.pdf", "page": 2},
        },
    ]

    # 2. Upsert chunks
    upserted_count = await pgvector_store.upsert_chunks(doc_id, proj_id, chunks)
    assert upserted_count == 2

    # 3. Perform similarity search
    results = await pgvector_store.search_similar(
        query_vector=dummy_vector,
        project_id=proj_id,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0]["document_id"] == doc_id
    assert results[0]["chunk_index"] == 0

    # 4. Delete document chunks
    deleted_count = await pgvector_store.delete_document_chunks(doc_id)
    assert deleted_count == 2

    # Verify empty after deletion
    post_del_results = await pgvector_store.search_similar(
        query_vector=dummy_vector,
        project_id=proj_id,
        top_k=2,
    )
    assert len(post_del_results) == 0


@pytest.mark.asyncio
async def test_neo4j_graph_store_operations():
    """Test Neo4j connection, constraint initialization, and MERGE operations."""
    connected = await neo4j_graph_store.check_connection()
    if not connected:
        pytest.skip("Local Neo4j instance is not running. Skipping live Neo4j integration test.")

    # 1. Initialize constraints
    await neo4j_graph_store.init_graph_store()

    # 2. Execute MERGE query
    query = """
    MERGE (u:User {id: $user_id})
    ON CREATE SET u.email = $email, u.created_at = datetime()
    RETURN u.id AS id, u.email AS email;
    """
    params = {"user_id": "test-user-999", "email": "unit_test@hayyuu.ai"}
    records = await neo4j_graph_store.execute_query(query, params)
    assert len(records) == 1
    assert records[0]["id"] == "test-user-999"
    assert records[0]["email"] == "unit_test@hayyuu.ai"

    # 3. Cleanup test user node
    cleanup_query = "MATCH (u:User {id: $user_id}) DETACH DELETE u;"
    await neo4j_graph_store.execute_query(cleanup_query, {"user_id": "test-user-999"})
