import os
import yaml
import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from app.infrastructure.mcp.server import create_sse_app, create_mcp_server
from app.infrastructure.mcp.client_manager import MCPClientManager


BACKEND_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_mcp_directives():
    """Verifies Dockerfile.mcp syntax, port exposure, healthchecks, and entrypoint."""
    dockerfile_path = BACKEND_ROOT / "Dockerfile.mcp"
    assert dockerfile_path.exists(), "Dockerfile.mcp must exist at backend root"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "EXPOSE 8001" in content, "Dockerfile.mcp must expose port 8001"
    assert "HEALTHCHECK" in content, "Dockerfile.mcp must define a container HEALTHCHECK"
    assert "http://localhost:8001/health" in content, "Healthcheck must target /health"
    assert "--transport" in content and "sse" in content, "Entrypoint must specify --transport sse"
    assert "8001" in content, "Entrypoint must specify port 8001"


def test_docker_compose_mcp_topology():
    """Verifies docker-compose.yaml includes the hayyuu_mcp service and wires it to web and worker."""
    compose_path = BACKEND_ROOT / "docker-compose.yaml"
    assert compose_path.exists(), "docker-compose.yaml must exist at backend root"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})

    # 1. Verify 'mcp' service
    assert "mcp" in services, "'mcp' service must be declared in docker-compose.yaml"
    mcp_svc = services["mcp"]
    assert mcp_svc.get("container_name") == "hayyuu_mcp"
    assert "8001:8001" in mcp_svc.get("ports", [])
    assert "healthcheck" in mcp_svc, "'mcp' service must have a healthcheck configured"
    assert mcp_svc.get("depends_on", {}).get("postgres", {}).get("condition") == "service_healthy"
    assert mcp_svc.get("depends_on", {}).get("redis", {}).get("condition") == "service_healthy"

    # 2. Verify 'web' service dependencies and environment override
    assert "web" in services
    web_svc = services["web"]
    web_env = web_svc.get("environment", {})
    assert web_env.get("HAYYUU_MCP_URL") == "http://mcp:8001/sse"
    assert "mcp" in web_svc.get("depends_on", {})
    assert web_svc["depends_on"]["mcp"].get("condition") == "service_healthy"

    # 3. Verify 'worker' service dependencies and environment override
    assert "worker" in services
    worker_svc = services["worker"]
    worker_env = worker_svc.get("environment", {})
    assert worker_env.get("HAYYUU_MCP_URL") == "http://mcp:8001/sse"
    assert "mcp" in worker_svc.get("depends_on", {})
    assert worker_svc["depends_on"]["mcp"].get("condition") == "service_healthy"


@pytest.mark.asyncio
async def test_client_manager_environment_override():
    """Verifies that MCPClientManager automatically uses HAYYUU_MCP_URL when set in container."""
    with patch.dict(os.environ, {"HAYYUU_MCP_URL": "http://mcp:8001/sse"}):
        mgr = MCPClientManager()
        with patch.object(mgr, "connect_server", new_callable=AsyncMock) as mock_connect:
            await mgr.connect_all()
            # Must connect to 'hayyuu-mcp' with transport 'sse' and url 'http://mcp:8001/sse'
            mock_connect.assert_any_call("hayyuu-mcp", {
                "transport": "sse",
                "url": "http://mcp:8001/sse"
            })


@pytest.mark.asyncio
async def test_docker_health_endpoint_contract():
    """Verifies that the /health endpoint satisfies the Docker healthcheck contract."""
    mcp_instance = create_mcp_server(name="test-health-contract")
    app = create_sse_app(server=mcp_instance)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8001") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "healthy"
        assert payload["service"] == "hayyuu-mcp"
        assert payload["version"] == "2.0.0"
        assert isinstance(payload["tools_count"], int)
        assert payload["tools_count"] >= 30
