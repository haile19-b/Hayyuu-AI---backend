import base64
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import Response

from app.infrastructure.mcp.server import create_mcp_server
from app.infrastructure.mcp.tools.github_tools import get_github_auth_headers


@pytest.fixture
def mcp_server():
    """Provides a fresh MCPServer instance configured with all Phase 1 & 2 capabilities."""
    return create_mcp_server(name="test-hayyuu-mcp-github")


ALL_21_GITHUB_TOOLS = [
    # 1. Files & Code
    "get_file_contents",
    "search_code",
    # 2. Issues & Discussions
    "list_issues",
    "get_issue",
    "create_issue",
    "update_issue",
    "add_issue_comment",
    # 3. Pull Requests & Reviews
    "list_pull_requests",
    "get_pull_request",
    "create_pull_request",
    "update_pull_request",
    "merge_pull_request",
    "create_pull_request_review",
    # 4. Branches & Commits
    "list_branches",
    "get_branch",
    "create_branch",
    "list_commits",
    "get_commit",
    # 5. Repositories
    "search_repositories",
    "create_repository",
    "fork_repository",
]


@pytest.mark.asyncio
async def test_all_21_github_tools_discovered(mcp_server):
    """Verifies that all 21 requested GitHub tools plus backward-compatible aliases are registered."""
    tools = await mcp_server.list_tools()
    tool_names = {t.name for t in tools}

    missing_tools = [tool for tool in ALL_21_GITHUB_TOOLS if tool not in tool_names]
    assert not missing_tools, f"Missing GitHub tools: {missing_tools}"

    # Also verify backward compatibility alias exists
    assert "create_github_issue" in tool_names


@pytest.mark.asyncio
async def test_github_prompts_discovered(mcp_server):
    """Verifies that GitHub-specific MCP prompts are registered and render properly."""
    prompts = await mcp_server.list_prompts()
    prompt_names = {p.name for p in prompts}

    assert "github_pr_review_template" in prompt_names
    assert "github_issue_spec_template" in prompt_names

    # Test PR review prompt rendering
    pr_prompt = await mcp_server.get_prompt(
        "github_pr_review_template",
        {"project_id": "proj-abc", "repo": "octocat/hello-world", "pull_number": 42}
    )
    pr_text = pr_prompt.messages[0].content.text
    assert "Pull Request #42" in pr_text
    assert "octocat/hello-world" in pr_text
    assert "proj-abc" in pr_text

    # Test issue spec prompt rendering
    issue_prompt = await mcp_server.get_prompt(
        "github_issue_spec_template",
        {"project_id": "proj-abc", "repo": "octocat/hello-world", "title": "Add OAuth2"}
    )
    issue_text = issue_prompt.messages[0].content.text
    assert "Add OAuth2" in issue_text
    assert "octocat/hello-world" in issue_text


@pytest.mark.asyncio
async def test_get_github_auth_headers_app_token_success():
    """Verifies dynamic token resolution uses github_service.get_token_for_project."""
    with patch("app.infrastructure.mcp.tools.github_tools.github_service.get_token_for_project", new_callable=AsyncMock) as mock_get_token:
        mock_get_token.return_value = "ghs_mock_installation_token_123"

        headers = await get_github_auth_headers("project-uuid-1")
        assert headers["Authorization"] == "Bearer ghs_mock_installation_token_123"
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"
        mock_get_token.assert_awaited_once_with("project-uuid-1")


@pytest.mark.asyncio
async def test_get_github_auth_headers_fallback_to_user_token():
    """Verifies fallback to owner.githubToken when GitHub App is not installed."""
    mock_project = MagicMock()
    mock_project.owner = MagicMock()
    mock_project.owner.githubToken = "gho_mock_user_oauth_token_456"

    mock_project_action = AsyncMock()
    mock_project_action.find_first.return_value = mock_project

    with patch("app.infrastructure.mcp.tools.github_tools.github_service.get_token_for_project", side_effect=Exception("App not installed")), \
         patch("app.infrastructure.mcp.tools.github_tools.prisma.project", mock_project_action):

        headers = await get_github_auth_headers("project-uuid-2")
        assert headers["Authorization"] == "Bearer gho_mock_user_oauth_token_456"


@pytest.mark.asyncio
async def test_tool_get_file_contents_base64_decoding(mcp_server):
    """Verifies get_file_contents decodes base64 text files correctly."""
    raw_code = "print('Hello, Hayyuu!')\n"
    encoded = base64.b64encode(raw_code.encode("utf-8")).decode("utf-8")

    mock_resp = Response(
        status_code=200,
        json={"name": "main.py", "path": "src/main.py", "encoding": "base64", "content": encoded, "size": len(raw_code)}
    )

    with patch("app.infrastructure.mcp.tools.github_tools.get_github_auth_headers", new_callable=AsyncMock) as mock_auth, \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_http_get:
        mock_auth.return_value = {"Authorization": "Bearer test"}
        mock_http_get.return_value = mock_resp

        result = await mcp_server.call_tool(
            "get_file_contents",
            {"project_id": "pid", "repo": "octocat/hello-world", "path": "src/main.py"}
        )
        assert result.content[0].text == raw_code


@pytest.mark.asyncio
async def test_tool_create_issue_and_alias(mcp_server):
    """Verifies create_issue and create_github_issue alias both post to GitHub API."""
    mock_resp = Response(
        status_code=201,
        json={"number": 105, "title": "Bug in parser", "html_url": "https://github.com/octocat/hello-world/issues/105"}
    )

    with patch("app.infrastructure.mcp.tools.github_tools.get_github_auth_headers", new_callable=AsyncMock) as mock_auth, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http_post:
        mock_auth.return_value = {"Authorization": "Bearer test"}
        mock_http_post.return_value = mock_resp

        # 1. Test create_issue
        res1 = await mcp_server.call_tool(
            "create_issue",
            {"project_id": "pid", "repo": "octocat/hello-world", "title": "Bug in parser", "body": "Details here"}
        )
        assert "Successfully created GitHub issue #105" in res1.content[0].text

        # 2. Test create_github_issue (legacy alias)
        res2 = await mcp_server.call_tool(
            "create_github_issue",
            {"project_id": "pid", "repo": "octocat/hello-world", "title": "Bug in parser", "body": "Details here"}
        )
        assert "Successfully created GitHub issue #105" in res2.content[0].text


@pytest.mark.asyncio
async def test_tool_pull_request_operations(mcp_server):
    """Verifies get_pull_request, merge_pull_request, and create_pull_request_review."""
    # 1. get_pull_request
    pr_resp = Response(
        status_code=200,
        json={
            "number": 12,
            "title": "Feature: Vector search",
            "state": "open",
            "merged": False,
            "mergeable": True,
            "head": {"ref": "feature-branch"},
            "base": {"ref": "main"},
            "user": {"login": "developer1"},
            "commits": 3,
            "changed_files": 5,
            "additions": 120,
            "deletions": 10,
            "html_url": "https://github.com/octocat/hello-world/pull/12",
            "body": "PR description"
        }
    )

    # 2. merge_pull_request
    merge_resp = Response(
        status_code=200,
        json={"sha": "6dcb09b5b57875f334f61aebed695e2e4193db5e", "merged": True, "message": "Pull Request successfully merged"}
    )

    with patch("app.infrastructure.mcp.tools.github_tools.get_github_auth_headers", new_callable=AsyncMock) as mock_auth, \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_http_get, \
         patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_http_put:
        mock_auth.return_value = {"Authorization": "Bearer test"}
        mock_http_get.return_value = pr_resp
        mock_http_put.return_value = merge_resp

        # Test get_pull_request
        pr_res = await mcp_server.call_tool(
            "get_pull_request",
            {"project_id": "pid", "repo": "octocat/hello-world", "pull_number": 12}
        )
        assert "Pull Request #12: Feature: Vector search" in pr_res.content[0].text
        assert "feature-branch -> main" in pr_res.content[0].text

        # Test merge_pull_request
        merge_res = await mcp_server.call_tool(
            "merge_pull_request",
            {"project_id": "pid", "repo": "octocat/hello-world", "pull_number": 12, "merge_method": "squash"}
        )
        assert "Successfully merged PR #12" in merge_res.content[0].text


@pytest.mark.asyncio
async def test_tool_branches_and_commits(mcp_server):
    """Verifies list_branches, create_branch, and list_commits."""
    branches_resp = Response(
        status_code=200,
        json=[
            {"name": "main", "commit": {"sha": "c0ffee1234567890"}, "protected": True},
            {"name": "dev", "commit": {"sha": "deadbeef12345678"}, "protected": False},
        ]
    )

    create_branch_resp = Response(
        status_code=201,
        json={"ref": "refs/heads/feature-xyz", "object": {"sha": "c0ffee1234567890"}}
    )

    with patch("app.infrastructure.mcp.tools.github_tools.get_github_auth_headers", new_callable=AsyncMock) as mock_auth, \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_http_get, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http_post:
        mock_auth.return_value = {"Authorization": "Bearer test"}
        mock_http_get.return_value = branches_resp
        mock_http_post.return_value = create_branch_resp

        # Test list_branches
        b_res = await mcp_server.call_tool("list_branches", {"project_id": "pid", "repo": "octocat/hello-world"})
        assert "- 'main'" in b_res.content[0].text
        assert "- 'dev'" in b_res.content[0].text

        # Test create_branch
        cb_res = await mcp_server.call_tool(
            "create_branch",
            {"project_id": "pid", "repo": "octocat/hello-world", "branch": "feature-xyz", "sha": "c0ffee1234567890"}
        )
        assert "Successfully created branch 'feature-xyz'" in cb_res.content[0].text


@pytest.mark.asyncio
async def test_tool_search_repositories(mcp_server):
    """Verifies search_repositories formats repositories list."""
    search_resp = Response(
        status_code=200,
        json={
            "items": [
                {
                    "full_name": "hayyuu-ai/backend",
                    "stargazers_count": 128,
                    "forks_count": 14,
                    "private": True,
                    "description": "Hayyuu AI Core Backend",
                    "html_url": "https://github.com/hayyuu-ai/backend"
                }
            ]
        }
    )

    with patch("app.infrastructure.mcp.tools.github_tools.get_github_auth_headers", new_callable=AsyncMock) as mock_auth, \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_http_get:
        mock_auth.return_value = {"Authorization": "Bearer test"}
        mock_http_get.return_value = search_resp

        res = await mcp_server.call_tool("search_repositories", {"project_id": "pid", "query": "hayyuu backend"})
        assert "hayyuu-ai/backend" in res.content[0].text
        assert "Hayyuu AI Core Backend" in res.content[0].text
