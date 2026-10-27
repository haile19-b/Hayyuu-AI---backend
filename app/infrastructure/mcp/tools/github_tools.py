import base64
import logging
from typing import List, Optional, Dict, Any
from mcp.server import MCPServer
import httpx

from app.core.database import prisma
from app.infrastructure.github.github_service import github_service

logger = logging.getLogger("uvicorn.error")

GITHUB_API_BASE = "https://api.github.com"


async def get_github_auth_headers(project_id: str) -> Dict[str, str]:
    """
    Resolves the active GitHub authentication token for the given project.
    Prioritizes GitHub App Installation Access Tokens via github_service,
    with fallback to project.owner.githubToken for legacy/personal OAuth tokens.
    """
    token: Optional[str] = None
    try:
        token = await github_service.get_token_for_project(project_id)
    except Exception as token_err:
        logger.warning(f"GitHub App token resolution for project '{project_id}' failed: {token_err}. Trying fallback...")
        project = await prisma.project.find_first(
            where={"id": project_id},
            include={"owner": True}
        )
        if project and project.owner and project.owner.githubToken:
            token = project.owner.githubToken
        else:
            raise ValueError(
                f"No active GitHub App installation or OAuth token available for project '{project_id}': {token_err}"
            )

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Hayyuu-AI-Agent",
    }


def _clean_repo(repo: str) -> str:
    """Strips leading/trailing slashes and github.com prefixes."""
    r = repo.strip()
    if r.startswith("https://github.com/"):
        r = r.replace("https://github.com/", "")
    elif r.startswith("http://github.com/"):
        r = r.replace("http://github.com/", "")
    return r.strip("/")


def register_github_tools(mcp: MCPServer) -> None:
    """Registers the complete 21-tool multi-tenant GitHub suite on the given MCPServer."""

    # =====================================================================
    # 1. Files & Code
    # =====================================================================

    @mcp.tool()
    async def get_file_contents(project_id: str, repo: str, path: str, ref: Optional[str] = None) -> str:
        """
        Retrieves the contents of a file or directory in a GitHub repository.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            path: Path to the file or directory.
            ref: Optional branch, tag, or commit SHA.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: get_file_contents] Fetching '{path}' in '{clean_r}' for project '{project_id}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/contents/{path.lstrip('/')}"
            params = {}
            if ref:
                params["ref"] = ref

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers, params=params)
                if res.status_code != 200:
                    return f"Failed to get file contents (HTTP {res.status_code}): {res.text}"

                data = res.json()
                if isinstance(data, list):
                    # Directory listing
                    entries = [f"- [{item.get('type')}] {item.get('name')} (path: {item.get('path')})" for item in data]
                    return f"Directory listing for '{path}':\n" + "\n".join(entries)

                encoding = data.get("encoding")
                raw_content = data.get("content", "")
                if encoding == "base64" and raw_content:
                    try:
                        decoded = base64.b64decode(raw_content).decode("utf-8")
                        return decoded
                    except UnicodeDecodeError:
                        return f"[Binary file: {data.get('size')} bytes, sha: {data.get('sha')}]"

                return raw_content or f"[File size: {data.get('size')} bytes, sha: {data.get('sha')}]"
        except Exception as e:
            logger.error(f"Error in get_file_contents: {e}")
            return f"Error fetching file contents: {str(e)}"

    @mcp.tool()
    async def search_code(project_id: str, query: str) -> str:
        """
        Searches code across GitHub repositories accessible to the project.

        Args:
            project_id: Active project UUID.
            query: The search query (e.g. 'def authenticate repo:owner/repo').
        """
        logger.info(f"[mcp: search_code] Searching code with query '{query}' for project '{project_id}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/search/code"
            params = {"q": query}

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers, params=params)
                if res.status_code != 200:
                    return f"Code search failed (HTTP {res.status_code}): {res.text}"

                items = res.json().get("items", [])
                if not items:
                    return f"No code results found for query '{query}'."

                results = [
                    f"- File: '{item.get('name')}' in repo '{item.get('repository', {}).get('full_name')}' (path: {item.get('path')}): {item.get('html_url')}"
                    for item in items[:15]
                ]
                return f"Found {len(items)} results (showing top {len(results)}):\n" + "\n".join(results)
        except Exception as e:
            logger.error(f"Error in search_code: {e}")
            return f"Error searching code: {str(e)}"

    # =====================================================================
    # 2. Issues & Discussions
    # =====================================================================

    @mcp.tool()
    async def list_issues(
        project_id: str,
        repo: str,
        state: str = "open",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 30,
        page: int = 1
    ) -> str:
        """
        Lists issues in a repository (excluding pull requests).

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            state: Filter by state: 'open', 'closed', or 'all'.
            sort: Sort field: 'created', 'updated', 'comments'.
            direction: Sort direction: 'asc' or 'desc'.
            per_page: Number of issues per page (max 100).
            page: Page number.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: list_issues] Listing issues for '{clean_r}' (state={state})")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/issues"
            params = {
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": min(max(per_page, 1), 100),
                "page": max(page, 1),
            }

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers, params=params)
                if res.status_code != 200:
                    return f"Failed to list issues (HTTP {res.status_code}): {res.text}"

                data = res.json()
                issues = [iss for iss in data if not iss.get("pull_request")]
                if not issues:
                    return f"No issues found in '{clean_r}' with state '{state}'."

                lines = []
                for iss in issues:
                    labels = [lbl.get("name") for lbl in iss.get("labels", []) if isinstance(lbl, dict)]
                    label_str = f" [labels: {', '.join(labels)}]" if labels else ""
                    lines.append(
                        f"- #{iss.get('number')}: '{iss.get('title')}' (state: {iss.get('state')}, author: {iss.get('user', {}).get('login')}){label_str} -> {iss.get('html_url')}"
                    )
                return f"Issues in '{clean_r}':\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Error in list_issues: {e}")
            return f"Error listing issues: {str(e)}"

    @mcp.tool()
    async def get_issue(project_id: str, repo: str, issue_number: int) -> str:
        """
        Retrieves full details of a specific issue.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            issue_number: Issue number.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: get_issue] Fetching issue #{issue_number} in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/issues/{issue_number}"

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers)
                if res.status_code != 200:
                    return f"Failed to get issue #{issue_number} (HTTP {res.status_code}): {res.text}"

                iss = res.json()
                labels = [lbl.get("name") for lbl in iss.get("labels", []) if isinstance(lbl, dict)]
                assignees = [a.get("login") for a in iss.get("assignees", []) if isinstance(a, dict)]

                return (
                    f"Issue #{iss.get('number')}: {iss.get('title')}\n"
                    f"- State: {iss.get('state')}\n"
                    f"- Author: {iss.get('user', {}).get('login')}\n"
                    f"- Assignees: {', '.join(assignees) if assignees else 'None'}\n"
                    f"- Labels: {', '.join(labels) if labels else 'None'}\n"
                    f"- Comments: {iss.get('comments', 0)}\n"
                    f"- URL: {iss.get('html_url')}\n"
                    f"- Created: {iss.get('created_at')}\n"
                    f"- Description:\n{iss.get('body') or 'No description provided.'}"
                )
        except Exception as e:
            logger.error(f"Error in get_issue: {e}")
            return f"Error getting issue: {str(e)}"

    @mcp.tool()
    async def create_issue(
        project_id: str,
        repo: str,
        title: str,
        body: str,
        assignees: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        milestone: Optional[int] = None
    ) -> str:
        """
        Creates a new GitHub issue in the specified repository.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            title: Issue title.
            body: Markdown issue description.
            assignees: Optional list of GitHub usernames to assign.
            labels: Optional list of label names.
            milestone: Optional milestone ID number.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: create_issue] Creating issue '{title}' in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/issues"
            payload: Dict[str, Any] = {"title": title, "body": body}
            if assignees:
                payload["assignees"] = assignees
            if labels:
                payload["labels"] = labels
            if milestone is not None:
                payload["milestone"] = milestone

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 201:
                    data = res.json()
                    return f"Successfully created GitHub issue #{data.get('number')}: {data.get('html_url')}"
                return f"Failed to create GitHub issue (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error creating issue: {e}")
            return f"Error creating GitHub issue: {str(e)}"

    # Backward compatibility alias
    @mcp.tool()
    async def create_github_issue(project_id: str, title: str, body: str, repo: str) -> str:
        """Alias for create_issue maintaining backward compatibility with early trial implementations."""
        return await create_issue(project_id=project_id, repo=repo, title=title, body=body)

    @mcp.tool()
    async def update_issue(
        project_id: str,
        repo: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        assignees: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        milestone: Optional[int] = None
    ) -> str:
        """
        Updates an existing issue (title, description, open/closed state, assignees, or labels).

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            issue_number: Issue number to update.
            title: Optional updated title.
            body: Optional updated markdown description.
            state: Optional updated state ('open' or 'closed').
            assignees: Optional updated list of assignees.
            labels: Optional updated list of labels.
            milestone: Optional updated milestone number.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: update_issue] Updating issue #{issue_number} in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/issues/{issue_number}"
            payload: Dict[str, Any] = {}
            if title is not None:
                payload["title"] = title
            if body is not None:
                payload["body"] = body
            if state is not None:
                payload["state"] = state
            if assignees is not None:
                payload["assignees"] = assignees
            if labels is not None:
                payload["labels"] = labels
            if milestone is not None:
                payload["milestone"] = milestone

            if not payload:
                return "No update parameters provided."

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.patch(url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return f"Successfully updated issue #{data.get('number')} (state: {data.get('state')}): {data.get('html_url')}"
                return f"Failed to update issue #{issue_number} (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in update_issue: {e}")
            return f"Error updating issue: {str(e)}"

    @mcp.tool()
    async def add_issue_comment(project_id: str, repo: str, issue_number: int, body: str) -> str:
        """
        Adds a new comment to an issue or pull request.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            issue_number: Issue or PR number.
            body: Markdown comment body.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: add_issue_comment] Commenting on #{issue_number} in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/issues/{issue_number}/comments"

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(url, headers=headers, json={"body": body})
                if res.status_code == 201:
                    data = res.json()
                    return f"Successfully added comment to #{issue_number}: {data.get('html_url')}"
                return f"Failed to add comment (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in add_issue_comment: {e}")
            return f"Error adding comment: {str(e)}"

    # =====================================================================
    # 3. Pull Requests & Reviews
    # =====================================================================

    @mcp.tool()
    async def list_pull_requests(
        project_id: str,
        repo: str,
        state: str = "open",
        head: Optional[str] = None,
        base: Optional[str] = None,
        sort: str = "created",
        direction: str = "desc",
        per_page: int = 30,
        page: int = 1
    ) -> str:
        """
        Lists pull requests in a repository.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            state: Filter by state: 'open', 'closed', or 'all'.
            head: Filter by head user/org and branch (e.g. 'user:branch').
            base: Filter by base branch.
            sort: Sort field: 'created', 'updated', 'popularity', 'long-running'.
            direction: Sort direction: 'asc' or 'desc'.
            per_page: Results per page (max 100).
            page: Page number.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: list_pull_requests] Listing PRs for '{clean_r}' (state={state})")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/pulls"
            params: Dict[str, Any] = {
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": min(max(per_page, 1), 100),
                "page": max(page, 1),
            }
            if head:
                params["head"] = head
            if base:
                params["base"] = base

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers, params=params)
                if res.status_code != 200:
                    return f"Failed to list pull requests (HTTP {res.status_code}): {res.text}"

                prs = res.json()
                if not prs:
                    return f"No pull requests found in '{clean_r}' matching the filters."

                lines = []
                for p in prs:
                    lines.append(
                        f"- PR #{p.get('number')}: '{p.get('title')}' [{p.get('head', {}).get('ref')} -> {p.get('base', {}).get('ref')}] (state: {p.get('state')}, author: {p.get('user', {}).get('login')}) -> {p.get('html_url')}"
                    )
                return f"Pull Requests in '{clean_r}':\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Error in list_pull_requests: {e}")
            return f"Error listing pull requests: {str(e)}"

    @mcp.tool()
    async def get_pull_request(project_id: str, repo: str, pull_number: int) -> str:
        """
        Retrieves detailed information about a specific pull request.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            pull_number: Pull request number.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: get_pull_request] Fetching PR #{pull_number} in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/pulls/{pull_number}"

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers)
                if res.status_code != 200:
                    return f"Failed to get PR #{pull_number} (HTTP {res.status_code}): {res.text}"

                p = res.json()
                return (
                    f"Pull Request #{p.get('number')}: {p.get('title')}\n"
                    f"- State: {p.get('state')} (Merged: {p.get('merged', False)}, Mergeable: {p.get('mergeable')})\n"
                    f"- Branches: {p.get('head', {}).get('ref')} -> {p.get('base', {}).get('ref')}\n"
                    f"- Author: {p.get('user', {}).get('login')}\n"
                    f"- Commits: {p.get('commits', 0)}, Changed Files: {p.get('changed_files', 0)}\n"
                    f"- Additions: +{p.get('additions', 0)}, Deletions: -{p.get('deletions', 0)}\n"
                    f"- URL: {p.get('html_url')}\n"
                    f"- Description:\n{p.get('body') or 'No description provided.'}"
                )
        except Exception as e:
            logger.error(f"Error in get_pull_request: {e}")
            return f"Error getting pull request: {str(e)}"

    @mcp.tool()
    async def create_pull_request(
        project_id: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        draft: bool = False
    ) -> str:
        """
        Creates a new pull request in a repository.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            title: Title of the pull request.
            head: The name of the branch where your changes are implemented.
            base: The name of the branch you want your changes pulled into.
            body: Optional pull request description.
            draft: Whether to create the pull request as a draft.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: create_pull_request] Creating PR '{title}' [{head} -> {base}] in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/pulls"
            payload = {
                "title": title,
                "head": head,
                "base": base,
                "body": body or "",
                "draft": draft,
            }

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 201:
                    data = res.json()
                    return f"Successfully created PR #{data.get('number')}: {data.get('html_url')}"
                return f"Failed to create PR (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in create_pull_request: {e}")
            return f"Error creating pull request: {str(e)}"

    @mcp.tool()
    async def update_pull_request(
        project_id: str,
        repo: str,
        pull_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        base: Optional[str] = None
    ) -> str:
        """
        Updates an existing pull request (title, description, state, or target branch).

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            pull_number: Pull request number.
            title: Optional updated title.
            body: Optional updated body.
            state: Optional updated state ('open' or 'closed').
            base: Optional updated base branch.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: update_pull_request] Updating PR #{pull_number} in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/pulls/{pull_number}"
            payload: Dict[str, Any] = {}
            if title is not None:
                payload["title"] = title
            if body is not None:
                payload["body"] = body
            if state is not None:
                payload["state"] = state
            if base is not None:
                payload["base"] = base

            if not payload:
                return "No update parameters provided."

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.patch(url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return f"Successfully updated PR #{data.get('number')} (state: {data.get('state')}): {data.get('html_url')}"
                return f"Failed to update PR #{pull_number} (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in update_pull_request: {e}")
            return f"Error updating pull request: {str(e)}"

    @mcp.tool()
    async def merge_pull_request(
        project_id: str,
        repo: str,
        pull_number: int,
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
        merge_method: str = "merge"
    ) -> str:
        """
        Merges a pull request into the base branch.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            pull_number: Pull request number.
            commit_title: Optional title for the merge commit.
            commit_message: Optional extra detail for the merge commit.
            merge_method: Merge method: 'merge', 'squash', or 'rebase'. Default is 'merge'.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: merge_pull_request] Merging PR #{pull_number} via {merge_method} in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/pulls/{pull_number}/merge"
            payload: Dict[str, Any] = {"merge_method": merge_method}
            if commit_title:
                payload["commit_title"] = commit_title
            if commit_message:
                payload["commit_message"] = commit_message

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.put(url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return f"Successfully merged PR #{pull_number}: {data.get('message')} (SHA: {data.get('sha')})"
                return f"Failed to merge PR #{pull_number} (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in merge_pull_request: {e}")
            return f"Error merging pull request: {str(e)}"

    @mcp.tool()
    async def create_pull_request_review(
        project_id: str,
        repo: str,
        pull_number: int,
        body: str,
        event: str = "COMMENT",
        comments: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Submits a code review on a pull request.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            pull_number: Pull request number.
            body: The review summary message.
            event: Review verdict: 'APPROVE', 'REQUEST_CHANGES', or 'COMMENT'.
            comments: Optional line-level review comments: [{'path': str, 'position': int, 'body': str}].
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: create_pull_request_review] Submitting {event} review on PR #{pull_number} in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/pulls/{pull_number}/reviews"
            payload: Dict[str, Any] = {
                "body": body,
                "event": event.upper(),
            }
            if comments:
                payload["comments"] = comments

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return f"Successfully submitted PR review (ID: {data.get('id')}, state: {data.get('state')}): {data.get('html_url')}"
                return f"Failed to create review (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in create_pull_request_review: {e}")
            return f"Error creating pull request review: {str(e)}"

    # =====================================================================
    # 4. Branches & Commits
    # =====================================================================

    @mcp.tool()
    async def list_branches(
        project_id: str,
        repo: str,
        protected: Optional[bool] = None,
        per_page: int = 30,
        page: int = 1
    ) -> str:
        """
        Lists branches in a repository.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            protected: Optional filter for protected branches.
            per_page: Results per page (max 100).
            page: Page number.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: list_branches] Listing branches for '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/branches"
            params: Dict[str, Any] = {
                "per_page": min(max(per_page, 1), 100),
                "page": max(page, 1),
            }
            if protected is not None:
                params["protected"] = "true" if protected else "false"

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers, params=params)
                if res.status_code != 200:
                    return f"Failed to list branches (HTTP {res.status_code}): {res.text}"

                branches = res.json()
                if not branches:
                    return f"No branches found in '{clean_r}'."

                lines = [
                    f"- '{b.get('name')}' (SHA: {b.get('commit', {}).get('sha')[:8]}, protected: {b.get('protected', False)})"
                    for b in branches
                ]
                return f"Branches in '{clean_r}':\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Error in list_branches: {e}")
            return f"Error listing branches: {str(e)}"

    @mcp.tool()
    async def get_branch(project_id: str, repo: str, branch: str) -> str:
        """
        Retrieves detailed information about a specific branch.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            branch: Name of the branch.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: get_branch] Getting branch '{branch}' in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/branches/{branch}"

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers)
                if res.status_code != 200:
                    return f"Failed to get branch '{branch}' (HTTP {res.status_code}): {res.text}"

                b = res.json()
                commit = b.get("commit", {})
                return (
                    f"Branch '{b.get('name')}':\n"
                    f"- Protected: {b.get('protected', False)}\n"
                    f"- Latest Commit SHA: {commit.get('sha')}\n"
                    f"- Commit Author: {commit.get('commit', {}).get('author', {}).get('name')}\n"
                    f"- Commit Message: {commit.get('commit', {}).get('message')}"
                )
        except Exception as e:
            logger.error(f"Error in get_branch: {e}")
            return f"Error getting branch: {str(e)}"

    @mcp.tool()
    async def create_branch(project_id: str, repo: str, branch: str, sha: str) -> str:
        """
        Creates a new branch pointing to an existing commit SHA.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            branch: Name of the new branch (without 'refs/heads/').
            sha: Commit SHA to base the branch on.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: create_branch] Creating branch '{branch}' from {sha[:8]} in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/git/refs"
            payload = {
                "ref": f"refs/heads/{branch}",
                "sha": sha,
            }

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 201:
                    data = res.json()
                    return f"Successfully created branch '{branch}' pointing to {sha[:8]} (ref: {data.get('ref')})."
                return f"Failed to create branch '{branch}' (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in create_branch: {e}")
            return f"Error creating branch: {str(e)}"

    @mcp.tool()
    async def list_commits(
        project_id: str,
        repo: str,
        sha: Optional[str] = None,
        path: Optional[str] = None,
        author: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        per_page: int = 30,
        page: int = 1
    ) -> str:
        """
        Lists commits in a repository.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            sha: SHA or branch to start listing commits from.
            path: Only commits containing this file path.
            author: GitHub login or email address to filter by.
            since: ISO 8601 timestamp string (e.g. 2026-01-01T00:00:00Z).
            until: ISO 8601 timestamp string.
            per_page: Results per page (max 100).
            page: Page number.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: list_commits] Listing commits for '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/commits"
            params: Dict[str, Any] = {
                "per_page": min(max(per_page, 1), 100),
                "page": max(page, 1),
            }
            if sha:
                params["sha"] = sha
            if path:
                params["path"] = path
            if author:
                params["author"] = author
            if since:
                params["since"] = since
            if until:
                params["until"] = until

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers, params=params)
                if res.status_code != 200:
                    return f"Failed to list commits (HTTP {res.status_code}): {res.text}"

                commits = res.json()
                if not commits:
                    return f"No commits found in '{clean_r}' matching the filters."

                lines = []
                for c in commits:
                    c_info = c.get("commit", {})
                    author_name = c_info.get("author", {}).get("name", "Unknown")
                    first_line = (c_info.get("message") or "").splitlines()[0] if c_info.get("message") else "No message"
                    lines.append(f"- [{c.get('sha')[:8]}] {first_line} (author: {author_name}) -> {c.get('html_url')}")
                return f"Commits in '{clean_r}':\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Error in list_commits: {e}")
            return f"Error listing commits: {str(e)}"

    @mcp.tool()
    async def get_commit(project_id: str, repo: str, ref: str) -> str:
        """
        Retrieves detailed information about a specific commit, including diff stats.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            ref: The commit SHA or reference.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: get_commit] Getting commit '{ref}' in '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/commits/{ref}"

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers)
                if res.status_code != 200:
                    return f"Failed to get commit '{ref}' (HTTP {res.status_code}): {res.text}"

                c = res.json()
                c_info = c.get("commit", {})
                stats = c.get("stats", {})
                files = c.get("files", [])

                file_summary = [f"- {f.get('status')}: {f.get('filename')} (+{f.get('additions', 0)}, -{f.get('deletions', 0)})" for f in files[:20]]
                return (
                    f"Commit {c.get('sha')}:\n"
                    f"- Author: {c_info.get('author', {}).get('name')} <{c_info.get('author', {}).get('email')}>\n"
                    f"- Date: {c_info.get('author', {}).get('date')}\n"
                    f"- Message: {c_info.get('message')}\n"
                    f"- Stats: Total {stats.get('total', 0)} (Additions: +{stats.get('additions', 0)}, Deletions: -{stats.get('deletions', 0)})\n"
                    f"- Files Changed ({len(files)}):\n" + "\n".join(file_summary)
                )
        except Exception as e:
            logger.error(f"Error in get_commit: {e}")
            return f"Error getting commit: {str(e)}"

    # =====================================================================
    # 5. Repositories
    # =====================================================================

    @mcp.tool()
    async def search_repositories(
        project_id: str,
        query: str,
        sort: Optional[str] = None,
        order: str = "desc",
        per_page: int = 30,
        page: int = 1
    ) -> str:
        """
        Searches GitHub repositories.

        Args:
            project_id: Active project UUID.
            query: The search keywords and qualifiers (e.g. 'topic:machine-learning org:my-org').
            sort: Sort field: 'stars', 'forks', 'help-wanted-issues', 'updated'.
            order: 'asc' or 'desc'.
            per_page: Results per page (max 100).
            page: Page number.
        """
        logger.info(f"[mcp: search_repositories] Searching repos with query '{query}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/search/repositories"
            params: Dict[str, Any] = {
                "q": query,
                "order": order,
                "per_page": min(max(per_page, 1), 100),
                "page": max(page, 1),
            }
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.get(url, headers=headers, params=params)
                if res.status_code != 200:
                    return f"Search repositories failed (HTTP {res.status_code}): {res.text}"

                items = res.json().get("items", [])
                if not items:
                    return f"No repositories found for query '{query}'."

                lines = [
                    f"- {r.get('full_name')} (stars: {r.get('stargazers_count')}, forks: {r.get('forks_count')}, private: {r.get('private')}): {r.get('description') or 'No description'} -> {r.get('html_url')}"
                    for r in items[:15]
                ]
                return f"Found {len(items)} repositories:\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Error in search_repositories: {e}")
            return f"Error searching repositories: {str(e)}"

    @mcp.tool()
    async def create_repository(
        project_id: str,
        name: str,
        description: Optional[str] = None,
        private: bool = False,
        auto_init: bool = True
    ) -> str:
        """
        Creates a new GitHub repository for the authenticated user/app.

        Args:
            project_id: Active project UUID.
            name: Repository name.
            description: Optional short description.
            private: Whether the repository should be private.
            auto_init: Whether to initialize with an empty README.
        """
        logger.info(f"[mcp: create_repository] Creating repository '{name}' (private={private})")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/user/repos"
            payload: Dict[str, Any] = {
                "name": name,
                "description": description or "",
                "private": private,
                "auto_init": auto_init,
            }

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code in (200, 201):
                    data = res.json()
                    return f"Successfully created repository '{data.get('full_name')}': {data.get('html_url')}"
                return f"Failed to create repository (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in create_repository: {e}")
            return f"Error creating repository: {str(e)}"

    @mcp.tool()
    async def fork_repository(project_id: str, repo: str, organization: Optional[str] = None) -> str:
        """
        Forks a repository into the authenticated user's account or organization.

        Args:
            project_id: Active project UUID.
            repo: Target repository to fork (e.g. 'octocat/hello-world').
            organization: Optional organization login to fork into.
        """
        clean_r = _clean_repo(repo)
        logger.info(f"[mcp: fork_repository] Forking repository '{clean_r}'")
        try:
            headers = await get_github_auth_headers(project_id)
            url = f"{GITHUB_API_BASE}/repos/{clean_r}/forks"
            payload: Dict[str, Any] = {}
            if organization:
                payload["organization"] = organization

            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code in (200, 202):
                    data = res.json()
                    return f"Successfully initiated fork of '{clean_r}' -> '{data.get('full_name')}': {data.get('html_url')}"
                return f"Failed to fork repository (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            logger.error(f"Error in fork_repository: {e}")
            return f"Error forking repository: {str(e)}"
