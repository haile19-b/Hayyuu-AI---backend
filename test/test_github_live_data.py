import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import jwt

from app.main import app
from app.infrastructure.github.github_service import (
    GitHubService,
    GitHubAppError,
)
from app.domain.entities.auth.jwt import sign_access_token


@pytest.fixture(scope="session")
def rsa_keypair():
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
    # Pre-seed token cache so mock calls avoid JWT exchange
    service._installation_tokens["inst-test-123"] = ("mock-access-token", time.time() + 3600)
    return service


@pytest.mark.asyncio
async def test_get_repository_branches(test_github_service):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"name": "main", "commit": {"sha": "1234567890abcdef"}, "protected": True},
        {"name": "develop", "commit": {"sha": "abcdef1234567890"}, "protected": False},
    ]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        branches = await test_github_service.get_repository_branches("inst-test-123", "hayyuu-org", "test-repo")

        assert len(branches) == 2
        assert branches[0]["name"] == "main"
        assert branches[0]["commitSha"] == "1234567890abcdef"
        assert branches[0]["protected"] is True
        assert branches[1]["name"] == "develop"


@pytest.mark.asyncio
async def test_get_repository_commits(test_github_service):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "sha": "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d",
            "commit": {
                "message": "feat: implement real-time github sync",
                "author": {"name": "Haile Dev", "date": "2026-10-14T12:00:00Z"},
            },
            "author": {"login": "haile-dev", "avatar_url": "https://github.com/haile.png"},
            "html_url": "https://github.com/hayyuu-org/test-repo/commit/7fd1a60b",
        }
    ]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        commits = await test_github_service.get_repository_commits(
            "inst-test-123", "hayyuu-org", "test-repo", branch="main", per_page=10
        )

        assert len(commits) == 1
        c = commits[0]
        assert c["hash"] == "7fd1a60"
        assert c["fullHash"] == "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d"
        assert c["message"] == "feat: implement real-time github sync"
        assert c["author"] == "haile-dev"
        assert c["authorAvatarUrl"] == "https://github.com/haile.png"
        assert c["date"] == "2026-10-14T12:00:00Z"


@pytest.mark.asyncio
async def test_get_repository_pull_requests(test_github_service):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "number": 42,
            "title": "Add multi-repo environment support",
            "state": "open",
            "merged_at": None,
            "user": {"login": "dev1", "avatar_url": "https://github.com/dev1.png"},
            "head": {"ref": "feature/multi-repo"},
            "base": {"ref": "main"},
            "draft": False,
            "created_at": "2026-10-14T10:00:00Z",
            "updated_at": "2026-10-14T11:00:00Z",
            "html_url": "https://github.com/hayyuu-org/test-repo/pull/42",
        },
        {
            "number": 40,
            "title": "Fix webhook parsing",
            "state": "closed",
            "merged_at": "2026-10-13T15:00:00Z",
            "user": {"login": "dev2"},
            "head": {"ref": "fix/webhook"},
            "base": {"ref": "main"},
            "draft": False,
            "created_at": "2026-10-13T14:00:00Z",
            "updated_at": "2026-10-13T15:00:00Z",
            "html_url": "https://github.com/hayyuu-org/test-repo/pull/40",
        },
    ]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        prs = await test_github_service.get_repository_pull_requests(
            "inst-test-123", "hayyuu-org", "test-repo", state="all"
        )

        assert len(prs) == 2
        assert prs[0]["number"] == 42
        assert prs[0]["status"] == "open"
        assert prs[0]["sourceBranch"] == "feature/multi-repo"
        assert prs[1]["number"] == 40
        assert prs[1]["status"] == "merged"


@pytest.mark.asyncio
async def test_get_repository_issues_filters_prs(test_github_service):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "number": 101,
            "title": "Database connection pool exhaustion on spike",
            "state": "open",
            "user": {"login": "reporter"},
            "labels": [{"name": "bug"}, {"name": "p0"}],
            "created_at": "2026-10-14T08:00:00Z",
            "updated_at": "2026-10-14T09:00:00Z",
            "comments": 3,
            "html_url": "https://github.com/hayyuu-org/test-repo/issues/101",
        },
        {
            # This is a pull request returned by GitHub issues endpoint
            "number": 102,
            "title": "PR that should be filtered out",
            "pull_request": {"url": "https://api.github.com/repos/hayyuu-org/test-repo/pulls/102"},
            "state": "open",
        },
    ]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        issues = await test_github_service.get_repository_issues(
            "inst-test-123", "hayyuu-org", "test-repo", state="open"
        )

        assert len(issues) == 1
        assert issues[0]["number"] == 101
        assert issues[0]["title"] == "Database connection pool exhaustion on spike"
        assert issues[0]["labels"] == ["bug", "p0"]
        assert issues[0]["commentsCount"] == 3


@pytest.mark.asyncio
async def test_get_repository_live_summary(test_github_service):
    branches_mock = [{"name": "main", "commitSha": "abc", "protected": True, "isDefault": False}]
    pulls_mock = [{"number": 1, "title": "P1", "status": "open", "state": "open"}]
    issues_mock = [{"number": 2, "title": "I1", "state": "open", "labels": []}]
    commits_mock = [{"hash": "abc", "fullHash": "abc1234", "message": "msg", "author": "a"}]

    with patch.object(test_github_service, "get_repository_branches", new_callable=AsyncMock) as mock_br, \
         patch.object(test_github_service, "get_repository_pull_requests", new_callable=AsyncMock) as mock_pr, \
         patch.object(test_github_service, "get_repository_issues", new_callable=AsyncMock) as mock_iss, \
         patch.object(test_github_service, "get_repository_commits", new_callable=AsyncMock) as mock_cm:

        mock_br.return_value = branches_mock
        mock_pr.return_value = pulls_mock
        mock_iss.return_value = issues_mock
        mock_cm.return_value = commits_mock

        summary = await test_github_service.get_repository_live_summary(
            "inst-test-123", "hayyuu-org", "test-repo"
        )

        assert summary["repoOwner"] == "hayyuu-org"
        assert summary["repoName"] == "test-repo"
        assert summary["activeBranch"] == "main"
        assert len(summary["commits"]) == 1
        assert len(summary["pullRequests"]) == 1
        assert len(summary["issues"]) == 1
        assert summary["openIssuesCount"] == 1
        assert summary["openPRsCount"] == 1
        assert summary["mergedPRsCount"] == 0


def test_live_data_route_unauthorized():
    client = TestClient(app)
    resp = client.get("/api/v1/projects/fake-proj-id/github/live-data")
    assert resp.status_code == 401


def test_live_data_route_success():
    client = TestClient(app)
    user_id = "user-test-777"
    project_id = "proj-test-888"

    token = sign_access_token({"id": user_id, "email": "test@example.com"})
    headers = {"Authorization": f"Bearer {token}"}

    mock_project = MagicMock()
    mock_project.id = project_id
    mock_project.userId = user_id
    mock_project.name = "Test Project"
    mock_project.githubInstallationId = "inst-999"
    mock_project.githubRepoOwner = "hayyuu-org"
    mock_project.githubRepoName = "sample-repo"

    mock_summary = {
        "repoOwner": "hayyuu-org",
        "repoName": "sample-repo",
        "repoUrl": "https://github.com/hayyuu-org/sample-repo",
        "activeBranch": "main",
        "branches": [{"name": "main", "commitSha": "abc1234", "protected": False, "isDefault": True}],
        "commits": [{"hash": "abc1234", "fullHash": "abc1234full", "message": "initial", "author": "dev"}],
        "pullRequests": [],
        "issues": [],
        "openIssuesCount": 0,
        "openPRsCount": 0,
        "mergedPRsCount": 0,
    }

    mock_prisma = MagicMock()
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)
    mock_prisma.projectrepository.find_many = AsyncMock(return_value=[])

    with patch("app.domain.entities.projects.github_route.prisma", mock_prisma), \
         patch("app.infrastructure.github.github_service.github_service.get_repository_live_summary", new_callable=AsyncMock) as mock_live:

        mock_live.return_value = mock_summary

        resp = client.get(f"/api/v1/projects/{project_id}/github/live-data", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["response"]["repoName"] == "sample-repo"
        assert data["response"]["activeBranch"] == "main"
        assert len(data["response"]["commits"]) == 1


def test_select_repo_exclusivity_conflict():
    client = TestClient(app)
    user_id = "user-test-777"
    project_id = "proj-test-1"

    token = sign_access_token({"id": user_id, "email": "test@example.com"})
    headers = {"Authorization": f"Bearer {token}"}

    mock_project = MagicMock()
    mock_project.id = project_id
    mock_project.userId = user_id
    mock_project.name = "Project 1"
    mock_project.githubInstallationId = "inst-999"

    # Mock other project claiming the repo
    mock_other_project = MagicMock()
    mock_other_project.id = "proj-test-2"
    mock_other_project.name = "Project 2"
    mock_rep = MagicMock()
    mock_rep.fullName = "hayyuu-org/exclusive-repo"
    mock_other_project.repositories = [mock_rep]
    mock_other_project.githubRepoOwner = "hayyuu-org"
    mock_other_project.githubRepoName = "exclusive-repo"

    mock_prisma = MagicMock()
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)
    mock_prisma.project.find_many = AsyncMock(return_value=[mock_other_project])

    with patch("app.domain.entities.projects.github_route.prisma", mock_prisma):
        resp = client.post(
            f"/api/v1/projects/{project_id}/github/select-repo",
            headers=headers,
            json={"owner": "hayyuu-org", "repo": "exclusive-repo"},
        )

        assert resp.status_code == 409
        data = resp.json()
        assert data["success"] is False
        assert "already assigned to project 'Project 2'" in data["error"]


def test_live_data_unassigned_repo_forbidden():
    client = TestClient(app)
    user_id = "user-test-777"
    project_id = "proj-test-1"

    token = sign_access_token({"id": user_id, "email": "test@example.com"})
    headers = {"Authorization": f"Bearer {token}"}

    mock_project = MagicMock()
    mock_project.id = project_id
    mock_project.userId = user_id
    mock_project.name = "Project 1"
    mock_project.githubInstallationId = "inst-999"
    mock_project.githubRepoOwner = "hayyuu-org"
    mock_project.githubRepoName = "my-assigned-repo"

    mock_prisma = MagicMock()
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)
    mock_prisma.projectrepository.find_many = AsyncMock(return_value=[])

    with patch("app.domain.entities.projects.github_route.prisma", mock_prisma):
        resp = client.get(
            f"/api/v1/projects/{project_id}/github/live-data?repo=unassigned-foreign-repo",
            headers=headers,
        )

        assert resp.status_code == 403
        data = resp.json()
        assert data["success"] is False
        assert "not assigned to this project" in data["error"]


@pytest.mark.asyncio
async def test_get_app_installations(test_github_service):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"id": 170996838, "account": {"login": "haile19-b"}}
    ]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        installs = await test_github_service.get_app_installations()
        assert len(installs) == 1
        assert installs[0]["id"] == 170996838


@pytest.mark.asyncio
async def test_get_repository_installation(test_github_service):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": 170996838, "account": {"login": "haile19-b"}}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        inst = await test_github_service.get_repository_installation("haile19-b", "my-todo")
        assert inst is not None
        assert inst["id"] == 170996838


def test_select_repo_auto_discovers_installation():
    client = TestClient(app)
    user_id = "user-test-auto"
    project_id = "proj-test-auto"

    token = sign_access_token({"id": user_id, "email": "test@example.com"})
    headers = {"Authorization": f"Bearer {token}"}

    mock_project = MagicMock()
    mock_project.id = project_id
    mock_project.userId = user_id
    mock_project.name = "My Project"
    mock_project.githubInstallationId = None
    mock_project.githubRepoOwner = None
    mock_project.githubRepoName = None
    mock_project.githubRepoUrl = None
    mock_project.githubConnectedAt = None

    mock_prisma = MagicMock()
    mock_prisma.project.find_unique = AsyncMock(return_value=mock_project)
    mock_prisma.project.find_first = AsyncMock(return_value=None)
    mock_prisma.project.find_many = AsyncMock(return_value=[mock_project])
    mock_prisma.project.update = AsyncMock(return_value=mock_project)
    mock_prisma.projectrepository.find_many = AsyncMock(return_value=[])
    mock_prisma.projectrepository.find_unique = AsyncMock(return_value=None)
    mock_prisma.projectrepository.update_many = AsyncMock()
    mock_prisma.projectrepository.create = AsyncMock()

    mock_repos = [
        {
            "id": 101,
            "name": "my-todo",
            "fullName": "haile19-b/my-todo",
            "owner": "haile19-b",
            "htmlUrl": "https://github.com/haile19-b/my-todo",
            "defaultBranch": "main",
            "private": False,
        }
    ]

    with patch("app.domain.entities.projects.github_route.prisma", mock_prisma):
        with patch("app.domain.entities.projects.github_route.github_service.get_repository_installation", new_callable=AsyncMock) as mock_repo_inst:
            with patch("app.domain.entities.projects.github_route.github_service.get_installation_repositories", new_callable=AsyncMock) as mock_get_repos:
                mock_repo_inst.return_value = {"id": 170996838}
                mock_get_repos.return_value = mock_repos

                resp = client.post(
                    f"/api/v1/projects/{project_id}/github/select-repo",
                    headers=headers,
                    json={"owner": "haile19-b", "repo": "my-todo"},
                )

                assert resp.status_code == 200
                data = resp.json()
                assert data["success"] is True


