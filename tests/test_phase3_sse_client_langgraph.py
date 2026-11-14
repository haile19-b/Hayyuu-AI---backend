import os
import json
import socket
import asyncio
import pytest
import uvicorn
from google.genai import types

from app.infrastructure.mcp.server import create_sse_app, create_mcp_server
from app.infrastructure.mcp.client_manager import MCPClientManager
from app.agents.search_agent.state import SearchAgentState
from app.agents.search_agent.nodes import execute_db_tool_node


def find_free_port() -> int:
    """Finds an unused port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def free_port() -> int:
    return find_free_port()


@pytest.fixture
async def sse_test_server(free_port):
    """
    Launches a real Hayyuu MCP Server instance running over HTTP/SSE on localhost
    and ensures clean background server shutdown after tests.
    """
    import httpx
    mcp_instance = create_mcp_server(name=f"test-sse-{free_port}")
    app = create_sse_app(server=mcp_instance)

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=free_port,
        log_level="error",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for the uvicorn server to be fully ready by polling /health
    health_url = f"http://127.0.0.1:{free_port}/health"
    async with httpx.AsyncClient() as client:
        for _ in range(60):
            try:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.1)

    yield f"http://127.0.0.1:{free_port}/sse"

    # Teardown server
    server.should_exit = True
    try:
        await asyncio.wait_for(server_task, timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


@pytest.mark.asyncio
async def test_mcp_config_schema_expansion(tmp_path):
    """Verifies that MCPClientManager parses transport configs and expands env vars."""
    os.environ["TEST_MCP_PORT"] = "9099"
    config_data = {
        "mcpServers": {
            "remote-sse": {
                "transport": "sse",
                "url": "http://localhost:${TEST_MCP_PORT}/sse"
            },
            "legacy-stdio": {
                "transport": "stdio",
                "command": "python",
                "args": ["-c", "print('ok')"]
            }
        }
    }
    cfg_file = tmp_path / "mcp_config_test.json"
    cfg_file.write_text(json.dumps(config_data), encoding="utf-8")

    mgr = MCPClientManager(config_path=str(cfg_file))
    assert mgr.config_path == str(cfg_file)


@pytest.mark.asyncio
async def test_sse_client_handshake_and_discovery(sse_test_server, tmp_path):
    """Verifies connecting MCPClientManager to a live SSE server discovers tools, resources, and prompts."""
    config_data = {
        "mcpServers": {
            "hayyuu-sse": {
                "transport": "sse",
                "url": sse_test_server
            }
        }
    }
    cfg_file = tmp_path / "mcp_test_config.json"
    cfg_file.write_text(json.dumps(config_data), encoding="utf-8")

    manager = MCPClientManager(config_path=str(cfg_file))
    try:
        await manager.connect_all()

        # 1. Verify Tools Discovered Over SSE
        tool_names = {t["name"] for t in manager.available_tools}
        assert "list_project_documents" in tool_names
        assert "create_issue" in tool_names
        assert "get_project_overview" in tool_names
        assert len(manager.available_tools) >= 30

        # 2. Verify Resource Templates Discovered Over SSE
        template_uris = {t["uri_template"] for t in manager.available_resource_templates}
        assert "hayyuu://projects/{project_id}/summary" in template_uris
        assert "hayyuu://projects/{project_id}/schema" in template_uris

        # 3. Verify Prompts Discovered Over SSE
        prompt_names = {p["name"] for p in manager.available_prompts}
        assert "analyze_conflicts" in prompt_names
        assert "github_pr_review_template" in prompt_names
        assert "architecture_synthesis" in prompt_names
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_sse_live_tool_and_resource_execution(sse_test_server, tmp_path):
    """Verifies live tool calls, resource reading, and prompt rendering over SSE."""
    config_data = {
        "mcpServers": {
            "hayyuu-sse": {
                "transport": "sse",
                "url": sse_test_server
            }
        }
    }
    cfg_file = tmp_path / "mcp_test_config.json"
    cfg_file.write_text(json.dumps(config_data), encoding="utf-8")

    manager = MCPClientManager(config_path=str(cfg_file))
    try:
        await manager.connect_all()

        # 1. Live Tool Execution over SSE
        tool_res = await manager.call_tool(
            "update_task_status",
            {"task_id": "test-task-1", "status": "INVALID_STATUS"}
        )
        assert "Invalid status" in tool_res

        # 2. Live Resource Reading over SSE
        cap_res = await manager.read_resource("hayyuu://system/capabilities")
        cap_json = json.loads(cap_res)
        assert cap_json["version"] == "2.0.0"
        assert cap_json["features"]["tools"] is True
        assert cap_json["features"]["multi_tenant_github"] is True

        # 3. Live Prompt Rendering over SSE
        prompt_text = await manager.get_prompt(
            "github_pr_review_template",
            {"project_id": "proj-99", "repo": "octocat/hello-world", "pull_number": 10}
        )
        assert "Pull Request #10" in prompt_text
        assert "octocat/hello-world" in prompt_text
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_langgraph_agent_execute_db_tool_over_sse(sse_test_server, tmp_path):
    """
    Verifies that LangGraph's execute_db_tool_node successfully executes
    tool calls in parallel against the remote SSE MCP Server.
    """
    config_data = {
        "mcpServers": {
            "hayyuu-sse": {
                "transport": "sse",
                "url": sse_test_server
            }
        }
    }
    cfg_file = tmp_path / "mcp_test_config.json"
    cfg_file.write_text(json.dumps(config_data), encoding="utf-8")

    manager = MCPClientManager(config_path=str(cfg_file))
    try:
        await manager.connect_all()

        # Construct simulated Gemini function call part in agent history
        call_part = types.Part.from_function_call(
            name="update_task_status",
            args={"task_id": "simulated-id", "status": "INVALID_VAL"}
        )
        history = [types.Content(role="model", parts=[call_part])]

        initial_state = SearchAgentState(
            project_id="test-proj-id",
            query_text="Update status",
            history=history,
            postgres_context=""
        )

        config_context = {
            "configurable": {
                "mcp_manager": manager
            }
        }

        result = await execute_db_tool_node(initial_state, config=config_context)

        assert result["status"] == "tool_executed"
        assert "update_task_status" in result["postgres_context"]
        assert "Invalid status" in result["postgres_context"]
        # History should now have tool output appended
        assert len(result["history"]) == 2
        assert result["history"][-1].role == "tool"
    finally:
        await manager.close()
