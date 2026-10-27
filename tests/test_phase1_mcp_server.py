import json
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from app.infrastructure.mcp.server import create_mcp_server, create_sse_app


@pytest.fixture
def mcp_server():
    """Provides a fresh MCPServer instance configured with all Phase 1 capabilities."""
    return create_mcp_server(name="test-hayyuu-mcp")


@pytest.mark.asyncio
async def test_mcp_server_tools_discovery(mcp_server):
    """Verifies that all DB, Graph, and Curated Backend tools are registered."""
    tools = await mcp_server.list_tools()
    tool_names = {t.name for t in tools}

    expected_tools = {
        # Core DB & Graph Tools
        "list_project_documents",
        "list_project_requirements",
        "list_project_tasks",
        "list_project_conflicts",
        "list_project_suggestions",
        "get_project_graph_summary",
        "traverse_project_subgraph",
        "create_github_issue",
        # Curated FastAPI Backend Tools
        "get_project_overview",
        "create_project_task",
        "update_task_status",
        "get_document_parsing_status",
    }

    missing = expected_tools - tool_names
    assert not missing, f"Missing expected tools: {missing}"
    assert len(tools) >= 12


@pytest.mark.asyncio
async def test_mcp_server_resources_discovery(mcp_server):
    """Verifies that project and system resources and templates are registered."""
    templates = await mcp_server.list_resource_templates()
    template_uris = {t.uri_template for t in templates}

    expected_templates = {
        "hayyuu://projects/{project_id}/summary",
        "hayyuu://projects/{project_id}/schema",
    }
    assert expected_templates.issubset(template_uris), f"Missing templates: {expected_templates - template_uris}"


@pytest.mark.asyncio
async def test_mcp_server_prompts_discovery(mcp_server):
    """Verifies that all specialized agent prompts are registered."""
    prompts = await mcp_server.list_prompts()
    prompt_names = {p.name for p in prompts}

    expected_prompts = {
        "analyze_conflicts",
        "task_breakdown_spec",
        "architecture_synthesis",
    }
    assert expected_prompts.issubset(prompt_names), f"Missing prompts: {expected_prompts - prompt_names}"


@pytest.mark.asyncio
async def test_system_capabilities_resource(mcp_server):
    """Verifies reading the system capabilities resource returns valid JSON."""
    result = await mcp_server.read_resource("hayyuu://system/capabilities")
    assert result and len(result) > 0
    content = result[0].content
    parsed = json.loads(content)

    assert parsed["name"] == "hayyuu-mcp-server"
    assert parsed["version"] == "2.0.0"
    assert parsed["features"]["tools"] is True
    assert parsed["features"]["resources"] is True
    assert parsed["features"]["prompts"] is True


@pytest.mark.asyncio
async def test_project_schema_resource(mcp_server):
    """Verifies reading project schema returns the complete data and graph ontology."""
    result = await mcp_server.read_resource("hayyuu://projects/test-uuid/schema")
    assert result and len(result) > 0
    content = result[0].content
    parsed = json.loads(content)

    assert parsed["project_id"] == "test-uuid"
    assert "Document" in parsed["entities"]
    assert "Requirement" in parsed["entities"]
    assert "Task" in parsed["entities"]
    assert len(parsed["neo4j_graph_relationships"]) > 0


@pytest.mark.asyncio
async def test_agent_prompts_rendering(mcp_server):
    """Verifies rendering prompts with parameters produces well-formed prompt messages."""
    # 1. analyze_conflicts
    res1 = await mcp_server.get_prompt("analyze_conflicts", {"project_id": "proj-101", "severity_filter": "HIGH"})
    assert res1.messages and len(res1.messages) > 0
    msg1 = res1.messages[0].content.text
    assert "proj-101" in msg1
    assert "HIGH" in msg1

    # 2. task_breakdown_spec
    res2 = await mcp_server.get_prompt("task_breakdown_spec", {"project_id": "proj-101", "requirement_id": "req-999"})
    msg2 = res2.messages[0].content.text
    assert "proj-101" in msg2
    assert "req-999" in msg2

    # 3. architecture_synthesis
    res3 = await mcp_server.get_prompt("architecture_synthesis", {"project_id": "proj-101"})
    msg3 = res3.messages[0].content.text
    assert "proj-101" in msg3
    assert "Architecture Blueprint" in msg3


@pytest.mark.asyncio
async def test_curated_backend_tool_update_task_validation(mcp_server):
    """Verifies that update_task_status validates input enum correctly."""
    result = await mcp_server.call_tool("update_task_status", {"task_id": "fake-id", "status": "INVALID_STATUS"})
    assert "Invalid status" in result.content[0].text


@pytest.mark.asyncio
async def test_curated_backend_tool_document_status_not_found(mcp_server):
    """Verifies get_document_parsing_status handles nonexistent document gracefully."""
    mock_doc = AsyncMock()
    mock_doc.find_unique.return_value = None
    with patch("app.infrastructure.mcp.tools.curated_backend_tools.prisma.document", mock_doc):
        result = await mcp_server.call_tool("get_document_parsing_status", {"document_id": "00000000-0000-0000-0000-000000000000"})
        assert "not found" in result.content[0].text



@pytest.mark.asyncio
async def test_sse_app_health_and_routes(mcp_server):
    """Verifies Starlette SSE ASGI application mounts /sse, /messages, and /health."""
    app = create_sse_app(server=mcp_server)
    routes = [r.path for r in app.routes]
    assert "/sse" in routes
    assert "/messages" in routes
    assert "/health" in routes

    # Test /health endpoint via ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "hayyuu-mcp"
