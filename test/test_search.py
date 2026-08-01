import sys
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.domain.entities.search.search_service import SearchService
from app.infrastructure.vector_store.pgvector import pgvector_store

@pytest.mark.asyncio
async def test_search_service_retrieval_and_synthesis_with_tool_calling():
    project_id = "test-proj-rag-123"
    doc_id = "test-doc-rag-123"
    query = "How many documents are in the auth module project?"

    # Ensure PGVector store is ready
    await pgvector_store.init_vector_store()

    # Pre-populate PGVector with some chunks for this project
    dummy_vector = [0.01] * 768
    chunks = [
        {
            "chunk_index": 0,
            "content": "The auth module project contains multiple requirements and design files.",
            "embedding": dummy_vector,
            "metadata": {"filename": "auth.py"},
        }
    ]
    await pgvector_store.upsert_chunks(doc_id, project_id, chunks)

    # Mock the external services
    with patch("app.domain.entities.search.search_service.gemini_embedder.embed_text") as mock_embed, \
         patch("app.domain.entities.search.search_service.neo4j_graph_store.execute_query", new_callable=AsyncMock) as mock_neo4j, \
         patch("app.domain.entities.search.search_service.genAI.models.generate_content") as mock_gen_content, \
         patch("app.domain.entities.search.search_service.prisma") as mock_prisma:

        mock_embed.return_value = dummy_vector

        # Mock database requirements/tasks/documents
        class MockRequirement:
            def __init__(self, id: str, title: str):
                self.id = id
                self.title = title

        class MockDocument:
            def __init__(self, id: str, name: str, fileType: str, sizeBytes: int, status: str):
                self.id = id
                self.name = name
                self.fileType = fileType
                self.sizeBytes = sizeBytes
                self.status = status

        mock_prisma.requirement.find_many = AsyncMock(return_value=[
            MockRequirement("da8636e0-2475-4d2d-9653-53d7e82b7db5", "Auth Module")
        ])
        mock_prisma.task.find_many = AsyncMock(return_value=[])
        mock_prisma.document.find_many = AsyncMock(return_value=[
            MockDocument("doc-999", "specification.pdf", "pdf", 1024, "INDEXED")
        ])

        # Mock Neo4j records
        mock_neo4j.side_effect = [
            # 1. Project nodes list query (for name matching)
            [
                {"id": "da8636e0-2475-4d2d-9653-53d7e82b7db5", "name": "Auth Module", "labels": ["Requirement"]}
            ],
            # 2. relationships query
            [
                {
                    "n": {"id": "da8636e0-2475-4d2d-9653-53d7e82b7db5", "name": "Auth Module", "description": "Validate users"},
                    "n_labels": ["Requirement"],
                    "rel_type": "DEPENDS_ON",
                    "m": {"id": "da8636e0-2475-4d2d-9653-53d7e82b7db6", "name": "Setup DB", "description": "Configure tables"},
                    "m_labels": ["Task"]
                }
            ],
            # 3. isolated nodes query
            [
                {
                    "n": {"id": "da8636e0-2475-4d2d-9653-53d7e82b7db5", "name": "Auth Module", "description": "Validate users"},
                    "n_labels": ["Requirement"]
                }
            ]
        ]

        # Define mocks representing Gemini function calling objects
        class MockFunctionCall:
            def __init__(self, name, args):
                self.name = name
                self.args = args

        class MockCandidate:
            def __init__(self, content):
                self.content = content

        class MockResponseWithTool:
            def __init__(self):
                self.function_calls = [MockFunctionCall("query_project_documents", {"project_id": project_id})]
                self.candidates = [MockCandidate("Mock calling database tool...")]
                self.text = ""

        class MockFinalResponse:
            def __init__(self):
                self.function_calls = []
                self.candidates = []
                self.text = "Synthesized answer: The project has 1 document: specification.pdf."

        # The generate_content call is made twice: first requests a function call, second turns tool response to text
        mock_gen_content.side_effect = [
            MockResponseWithTool(),
            MockFinalResponse()
        ]

        # Execute generate_rag_answer
        response = await SearchService.generate_rag_answer(
            project_id=project_id,
            query_text=query,
            document_id=doc_id,
            limit=1
        )

        # Assertions
        assert response.answer == "Synthesized answer: The project has 1 document: specification.pdf."
        assert len(response.sources) >= 1
        assert response.sources[0].content == "The auth module project contains multiple requirements and design files."
        
        # Verify graph node mapping was performed
        assert len(response.nodes) == 2
        node_ids = {n.id for n in response.nodes}
        assert "da8636e0-2475-4d2d-9653-53d7e82b7db5" in node_ids
        assert "da8636e0-2475-4d2d-9653-53d7e82b7db6" in node_ids

        # Verify that mock_prisma.document.find_many was called during the tool execution loop
        mock_prisma.document.find_many.assert_called_once_with(where={"projectId": project_id})

        # Cleanup pgvector chunks
        await pgvector_store.delete_document_chunks(doc_id)
