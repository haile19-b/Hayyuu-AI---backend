import sys
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.agents.search_agent.graph import search_agent_graph
from app.agents.search_agent.state import SearchAgentState

@pytest.mark.asyncio
async def test_search_agent_workflow():
    project_id = "test-proj-rag-123"
    doc_id = "test-doc-rag-123"
    query = "How many documents are in the auth module project?"

    dummy_vector = [0.01] * 768

    # Mock the external services at their definition sources
    with patch("app.infrastructure.ai.gemini_embedder.gemini_embedder.embed_text") as mock_embed, \
         patch("app.infrastructure.graph_store.neo4j.neo4j_graph_store.execute_query", new_callable=AsyncMock) as mock_neo4j, \
         patch("app.agents.search_agent.nodes.genAI.models.generate_content") as mock_gen_content, \
         patch("app.agents.search_agent.nodes.prisma") as mock_prisma_nodes, \
         patch("app.agents.search_agent.tools.prisma") as mock_prisma_tools:

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

        mock_prisma_nodes.requirement.find_many = AsyncMock(return_value=[
            MockRequirement("da8636e0-2475-4d2d-9653-53d7e82b7db5", "Auth Module")
        ])
        mock_prisma_nodes.task.find_many = AsyncMock(return_value=[])
        mock_prisma_tools.document.find_many = AsyncMock(return_value=[
            MockDocument("doc-999", "specification.pdf", "pdf", 1024, "INDEXED")
        ])

        # Mock Neo4j records
        mock_neo4j.side_effect = [
            # 1. Neo4j native vector search index query (in query_analysis_node)
            [
                {
                    "id": "chunk-123",
                    "document_id": doc_id,
                    "project_id": project_id,
                    "chunk_index": 0,
                    "content": "The auth module project contains multiple requirements and design files.",
                    "prev_content": None,
                    "next_content": None,
                    "metadata": '{"filename": "auth.py"}',
                    "similarity": 0.95
                }
            ],
            # 2. Project nodes list query (for name matching inside retrieve_hybrid_context/graph_retrieval_node)
            [
                {"id": "da8636e0-2475-4d2d-9653-53d7e82b7db5", "name": "Auth Module", "labels": ["Requirement"]}
            ],
            # 3. Neo4j relationships query (called inside graph_db_tool)
            [
                {
                    "n": {"id": "da8636e0-2475-4d2d-9653-53d7e82b7db5", "name": "Auth Module", "description": "Validate users"},
                    "n_labels": ["Requirement"],
                    "rel_type": "DEPENDS_ON",
                    "m": {"id": "da8636e0-2475-4d2d-9653-53d7e82b7db6", "name": "Setup DB", "description": "Configure tables"},
                    "m_labels": ["Task"]
                }
            ],
            # 4. Neo4j nodes query (called inside graph_db_tool)
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

        class MockPart:
            def __init__(self, function_call=None, text=""):
                self.function_call = function_call
                self.text = text

        class MockContent:
            def __init__(self, parts, role="model"):
                self.parts = parts
                self.role = role

        class MockCandidate:
            def __init__(self, content):
                self.content = content

        class MockResponseWithTool:
            def __init__(self):
                func_call = MockFunctionCall("relational_db_tool", {"project_id": project_id, "action": "list_documents"})
                self.function_calls = [func_call]
                self.candidates = [MockCandidate(MockContent([MockPart(function_call=func_call)]))]
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

        # Initialize the LangGraph Search Agent State
        state = SearchAgentState(
            project_id=project_id,
            query_text=query,
            document_id=doc_id,
            limit=1
        )

        # Invoke the LangGraph Search Agent Graph
        result_state = await search_agent_graph.ainvoke(state)

        # Assertions
        assert result_state.get("answer") == "Synthesized answer: The project has 1 document: specification.pdf."
        assert len(result_state.get("chunks", [])) >= 1
        assert result_state.get("chunks")[0].content == "The auth module project contains multiple requirements and design files."
        
        # Verify graph node mapping was performed
        assert len(result_state.get("graph_nodes", [])) == 2
        node_ids = {n.id for n in result_state.get("graph_nodes", [])}
        assert "da8636e0-2475-4d2d-9653-53d7e82b7db5" in node_ids
        assert "da8636e0-2475-4d2d-9653-53d7e82b7db6" in node_ids

        # Verify that mock_prisma_tools.document.find_many was called during the tool execution loop
        mock_prisma_tools.document.find_many.assert_called_once_with(where={"projectId": project_id})
