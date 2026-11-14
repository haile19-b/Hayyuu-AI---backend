import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import jwt

from app.main import app
from app.core.env import settings
from app.infrastructure.github.github_service import (
    GitHubService,
    GitHubAppError,
    GitHubConfigError,
    GitHubInstallationNotFoundError,
)
from app.domain.entities.auth.jwt import sign_access_token
from app.core.database import prisma


@pytest.fixture(scope="session")
def rsa_keypair():
    """Generates a temporary in-memory RSA keypair for testing JWT signatures."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


@pytest.fixture
def test_github_service(rsa_keypair):
    private_pem, _ = rsa_keypair
    service = GitHubService()
    service._get_private_key = lambda: private_pem
    service._get_app_id = lambda: "123456"
    return service


def test_jwt_generation(test_github_service, rsa_keypair):
    _, public_pem = rsa_keypair
    token = test_github_service.generate_app_jwt()
    assert token is not None

    decoded = jwt.decode(token, public_pem, algorithms=["RS256"])
    assert decoded["iss"] == "123456"
    assert "exp" in decoded
    assert "iat" in decoded


def test_state_creation_and_verification(test_github_service):
    project_id = "proj-uuid-123"
    user_id = "user-uuid-456"

    state = test_github_service.create_install_state(project_id, user_id)
    assert state is not None

    # Verification success
    assert test_github_service.verify_install_state(state, project_id, user_id) is True

    # Verification failure on mismatched project or user
    assert test_github_service.verify_install_state(state, "wrong-project", user_id) is False
    assert test_github_service.verify_install_state(state, project_id, "wrong-user") is False
    assert test_github_service.verify_install_state("invalid-state-token", project_id, user_id) is False


def test_install_url_generation(test_github_service):
    url, state = test_github_service.get_install_url("proj-abc", "user-xyz")
    assert "https://github.com/apps/hayyuu-ai/installations/new?state=" in url
    assert state in url


@pytest.mark.asyncio
async def test_installation_token_caching(test_github_service):
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 201
    mock_post_resp.json.return_value = {
        "token": "ghs_mock_installation_token_12345",
        "expires_at": "2030-01-01T00:00:00Z",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_post_resp

        # First call fetches from GitHub
        token1 = await test_github_service.get_installation_access_token("inst-999")
        assert token1 == "ghs_mock_installation_token_12345"
        assert mock_post.call_count == 1

        # Second call within validity should hit in-memory cache
        token2 = await test_github_service.get_installation_access_token("inst-999")
        assert token2 == "ghs_mock_installation_token_12345"
        assert mock_post.call_count == 1  # Not called again


@pytest.mark.asyncio
async def test_get_repositories(test_github_service):
    test_github_service._installation_tokens["inst-888"] = ("mock-token", time.time() + 3600)

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "total_count": 1,
        "repositories": [
            {
                "id": 101,
                "name": "sample-repo",
                "full_name": "hayyuu-org/sample-repo",
                "owner": {"login": "hayyuu-org"},
                "description": "Sample test repo",
                "html_url": "https://github.com/hayyuu-org/sample-repo",
                "default_branch": "main",
                "private": False,
                "updated_at": "2026-10-10T12:00:00Z",
            }
        ],
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_get_resp
        repos = await test_github_service.get_installation_repositories("inst-888")
        assert len(repos) == 1
        assert repos[0]["name"] == "sample-repo"
        assert repos[0]["owner"] == "hayyuu-org"


def test_api_routes_registered():
    from app.main import app
    route_paths = list(app.openapi().get("paths", {}).keys())
    expected = [
        "/api/v1/projects/{projectId}/github/install-url",
        "/api/v1/projects/{projectId}/github/connect",
        "/api/v1/projects/{projectId}/github/repositories",
        "/api/v1/projects/{projectId}/github/select-repo",
        "/api/v1/projects/{projectId}/github/status",
        "/api/v1/projects/{projectId}/github/disconnect",
    ]
    for exp in expected:
        assert exp in route_paths, f"Expected route {exp} not found in FastAPI application routes."

