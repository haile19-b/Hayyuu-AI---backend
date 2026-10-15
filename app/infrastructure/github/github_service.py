import time
import asyncio
import logging
from typing import Optional, Dict, Tuple, List, Any
import httpx
import jwt
from dateutil import parser as date_parser

from app.core.env import settings
from app.core.database import prisma

logger = logging.getLogger("uvicorn.error")


class GitHubAppError(Exception):
    """Base exception for GitHub App operations."""
    pass


class GitHubConfigError(GitHubAppError):
    """Raised when GitHub App configuration (ID or Private Key) is missing or invalid."""
    pass


class GitHubInstallationNotFoundError(GitHubAppError):
    """Raised when an installation ID is revoked, uninstalled, or not found."""
    pass


class GitHubRateLimitError(GitHubAppError):
    """Raised when GitHub API rate limits are hit."""
    pass


class GitHubService:
    """
    Infrastructure service for GitHub App integration.
    Handles JWT generation, installation access token generation,
    repository querying, and project token resolution for AI agents and MCP servers.
    """

    GITHUB_API_BASE = "https://api.github.com"

    def __init__(self):
        # Cache of (token_string, expiration_unix_timestamp) keyed by installation_id
        self._installation_tokens: Dict[str, Tuple[str, float]] = {}

    def _get_private_key(self) -> str:
        """Retrieves and validates the GitHub App private key."""
        key = settings.get_github_private_key()
        if not key:
            raise GitHubConfigError(
                "GitHub App Private Key is not configured. "
                "Please configure GITHUB_APP_PRIVATE_KEY_PATH or GITHUB_APP_PRIVATE_KEY in .env."
            )
        return key

    def _get_app_id(self) -> str:
        """Retrieves and validates the GitHub App ID."""
        import os
        app_id = settings.GITHUB_APP_ID or os.getenv("GITHUB_APP_ID") or "5300504"
        if not app_id or not str(app_id).strip():
            raise GitHubConfigError(
                "GitHub App ID is not configured. "
                "Please set GITHUB_APP_ID in .env."
            )
        return str(app_id).strip()

    def generate_app_jwt(self) -> str:
        """
        Generates a signed RS256 JWT for authenticating as the GitHub App.
        The JWT is valid for 10 minutes (with 60-second clock drift allowance).
        """
        app_id = self._get_app_id()
        private_key = self._get_private_key()

        now = int(time.time())
        payload = {
            # Issued at time, 60 seconds in the past to compensate for clock drift
            "iat": now - 60,
            # JWT expiration time (10 minute maximum allowable by GitHub)
            "exp": now + (10 * 60),
            # GitHub App ID
            "iss": app_id,
        }

        try:
            token = jwt.encode(payload, private_key, algorithm="RS256")
            return token
        except Exception as e:
            logger.error(f"Failed to sign GitHub App JWT: {e}")
            raise GitHubConfigError(f"Could not sign JWT with the provided GitHub App Private Key: {e}")

    async def get_installation_access_token(self, installation_id: str, force_refresh: bool = False) -> str:
        """
        Exchanges the GitHub App JWT for a scoped, ephemeral 1-hour Installation Access Token.
        Uses in-memory caching to reuse tokens until 5 minutes before expiration.
        """
        if not installation_id:
            raise GitHubAppError("Installation ID cannot be empty.")

        # Check in-memory cache first
        if not force_refresh and installation_id in self._installation_tokens:
            token, exp_time = self._installation_tokens[installation_id]
            # If token still has at least 5 minutes of validity, reuse it
            if exp_time - time.time() > 300:
                return token

        app_jwt = self.generate_app_jwt()
        url = f"{self.GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.post(url, headers=headers)
            except httpx.RequestError as exc:
                logger.error(f"Network error contacting GitHub API at {url}: {exc}")
                raise GitHubAppError(f"Network error connecting to GitHub: {exc}")

            if response.status_code == 404:
                # Installation was revoked or uninstalled
                self._installation_tokens.pop(installation_id, None)
                raise GitHubInstallationNotFoundError(
                    f"GitHub App installation '{installation_id}' was not found or has been revoked."
                )

            if response.status_code == 401:
                logger.error(f"GitHub App authentication rejected (401): {response.text}")
                raise GitHubAppError("GitHub App credentials rejected. Verify GITHUB_APP_ID and private key.")

            if response.status_code in (403, 429):
                retry_after = response.headers.get("retry-after", "60")
                logger.warning(f"GitHub rate limit reached: {response.text}")
                raise GitHubRateLimitError(f"GitHub API rate limit reached. Retry after {retry_after} seconds.")

            if response.status_code != 201:
                logger.error(f"GitHub installation token creation failed ({response.status_code}): {response.text}")
                raise GitHubAppError(f"Failed to obtain installation token (HTTP {response.status_code}): {response.text}")

            data = response.json()
            token_str = data["token"]
            expires_at_str = data.get("expires_at")

            # Parse expiration timestamp
            if expires_at_str:
                exp_dt = date_parser.parse(expires_at_str)
                exp_time = exp_dt.timestamp()
            else:
                exp_time = time.time() + 3600

            # Store in cache
            self._installation_tokens[installation_id] = (token_str, exp_time)
            return token_str

    async def get_installation_repositories(self, installation_id: str) -> List[Dict[str, Any]]:
        """
        Queries the GitHub API for all repositories authorized under the given installation.
        """
        token = await self.get_installation_access_token(installation_id)
        url = f"{self.GITHUB_API_BASE}/installation/repositories"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers)
            except httpx.RequestError as exc:
                raise GitHubAppError(f"Network error fetching repositories: {exc}")

            if response.status_code == 401:
                # Token may have expired, force refresh once
                token = await self.get_installation_access_token(installation_id, force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                response = await client.get(url, headers=headers)

            if response.status_code != 200:
                raise GitHubAppError(f"Failed to fetch repositories (HTTP {response.status_code}): {response.text}")

            data = response.json()
            repos = data.get("repositories", [])
            results = []
            for r in repos:
                results.append({
                    "id": r["id"],
                    "name": r["name"],
                    "fullName": r["full_name"],
                    "owner": r["owner"]["login"],
                    "description": r.get("description"),
                    "htmlUrl": r["html_url"],
                    "defaultBranch": r.get("default_branch", "main"),
                    "private": r.get("private", False),
                    "updatedAt": r.get("updated_at"),
                })
            return results

    async def get_app_installations(self) -> List[Dict[str, Any]]:
        """
        Retrieves all installations of the GitHub App using the App JWT.
        """
        jwt_token = self.generate_app_jwt()
        url = f"{self.GITHUB_API_BASE}/app/installations"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers)
            except httpx.RequestError as exc:
                raise GitHubAppError(f"Network error fetching installations: {exc}")

            if response.status_code != 200:
                raise GitHubAppError(f"Failed to fetch installations (HTTP {response.status_code}): {response.text}")

            return response.json()

    async def get_repository_installation(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """
        Checks if the GitHub App is installed on a specific repository using the App JWT.
        Returns the installation dictionary if found, or None if not installed (HTTP 404).
        """
        jwt_token = self.generate_app_jwt()
        url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/installation"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers)
            except httpx.RequestError as exc:
                raise GitHubAppError(f"Network error querying repository installation: {exc}")

            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return None
            if response.status_code in (403, 429):
                raise GitHubRateLimitError("GitHub API rate limit exceeded.")
            raise GitHubAppError(f"Error querying repository installation (HTTP {response.status_code}): {response.text}")

    async def get_repository_branches(
        self, installation_id: str, owner: str, repo: str
    ) -> List[Dict[str, Any]]:
        """
        Queries all branches for the specified repository using the installation access token.
        """
        token = await self.get_installation_access_token(installation_id)
        url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/branches"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        params = {"per_page": 100}

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers, params=params)
            except httpx.RequestError as exc:
                raise GitHubAppError(f"Network error fetching branches: {exc}")

            if response.status_code == 401:
                token = await self.get_installation_access_token(installation_id, force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                response = await client.get(url, headers=headers, params=params)

            if response.status_code == 404:
                raise GitHubAppError(f"Repository '{owner}/{repo}' not found on GitHub.")
            if response.status_code in (403, 429):
                retry_after = response.headers.get("retry-after", "60")
                raise GitHubRateLimitError(f"GitHub API rate limit reached. Retry after {retry_after} seconds.")
            if response.status_code != 200:
                raise GitHubAppError(f"Failed to fetch branches (HTTP {response.status_code}): {response.text}")

            data = response.json()
            results = []
            for b in data:
                commit_info = b.get("commit", {})
                results.append({
                    "name": b.get("name", ""),
                    "commitSha": commit_info.get("sha", ""),
                    "protected": b.get("protected", False),
                    "isDefault": False,
                })
            return results

    async def get_repository_commits(
        self,
        installation_id: str,
        owner: str,
        repo: str,
        branch: Optional[str] = None,
        per_page: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Queries recent commits for the specified repository and branch.
        """
        token = await self.get_installation_access_token(installation_id)
        url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        params: Dict[str, Any] = {"per_page": min(max(per_page, 1), 100)}
        if branch:
            params["sha"] = branch

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers, params=params)
            except httpx.RequestError as exc:
                raise GitHubAppError(f"Network error fetching commits: {exc}")

            if response.status_code == 401:
                token = await self.get_installation_access_token(installation_id, force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                response = await client.get(url, headers=headers, params=params)

            if response.status_code == 404:
                raise GitHubAppError(f"Repository or branch '{owner}/{repo}@{branch}' not found.")
            if response.status_code in (403, 429):
                retry_after = response.headers.get("retry-after", "60")
                raise GitHubRateLimitError(f"GitHub API rate limit reached. Retry after {retry_after} seconds.")
            if response.status_code != 200:
                raise GitHubAppError(f"Failed to fetch commits (HTTP {response.status_code}): {response.text}")

            data = response.json()
            results = []
            for c in data:
                commit_obj = c.get("commit", {})
                author_obj = commit_obj.get("author", {})
                gh_author = c.get("author") or {}
                sha = c.get("sha", "")
                results.append({
                    "hash": sha[:7] if sha else "",
                    "fullHash": sha,
                    "message": commit_obj.get("message", "").strip(),
                    "author": gh_author.get("login") or author_obj.get("name") or "Unknown",
                    "authorAvatarUrl": gh_author.get("avatar_url"),
                    "date": author_obj.get("date"),
                    "htmlUrl": c.get("html_url"),
                })
            return results

    async def get_repository_pull_requests(
        self,
        installation_id: str,
        owner: str,
        repo: str,
        state: str = "all",
        per_page: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Queries recent pull requests for the specified repository.
        """
        token = await self.get_installation_access_token(installation_id)
        url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/pulls"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        params = {
            "state": state,
            "per_page": min(max(per_page, 1), 100),
            "sort": "updated",
            "direction": "desc",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers, params=params)
            except httpx.RequestError as exc:
                raise GitHubAppError(f"Network error fetching pull requests: {exc}")

            if response.status_code == 401:
                token = await self.get_installation_access_token(installation_id, force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                response = await client.get(url, headers=headers, params=params)

            if response.status_code == 404:
                raise GitHubAppError(f"Repository '{owner}/{repo}' not found.")
            if response.status_code in (403, 429):
                retry_after = response.headers.get("retry-after", "60")
                raise GitHubRateLimitError(f"GitHub API rate limit reached. Retry after {retry_after} seconds.")
            if response.status_code != 200:
                raise GitHubAppError(f"Failed to fetch pull requests (HTTP {response.status_code}): {response.text}")

            data = response.json()
            results = []
            for pr in data:
                user_obj = pr.get("user") or {}
                head_obj = pr.get("head") or {}
                base_obj = pr.get("base") or {}
                
                # Derive status: 'merged', 'open', or 'closed'
                if pr.get("merged_at"):
                    status_val = "merged"
                else:
                    status_val = pr.get("state", "open")

                results.append({
                    "number": pr.get("number"),
                    "title": pr.get("title", ""),
                    "status": status_val,
                    "state": pr.get("state", "open"),
                    "author": user_obj.get("login", "Unknown"),
                    "authorAvatarUrl": user_obj.get("avatar_url"),
                    "createdAt": pr.get("created_at"),
                    "updatedAt": pr.get("updated_at"),
                    "mergedAt": pr.get("merged_at"),
                    "closedAt": pr.get("closed_at"),
                    "htmlUrl": pr.get("html_url"),
                    "draft": pr.get("draft", False),
                    "sourceBranch": head_obj.get("ref"),
                    "targetBranch": base_obj.get("ref"),
                })
            return results

    async def get_repository_issues(
        self,
        installation_id: str,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Queries repository issues, strictly filtering out pull requests.
        """
        token = await self.get_installation_access_token(installation_id)
        url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        params = {
            "state": state,
            "per_page": min(max(per_page, 1), 100),
            "sort": "updated",
            "direction": "desc",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers, params=params)
            except httpx.RequestError as exc:
                raise GitHubAppError(f"Network error fetching issues: {exc}")

            if response.status_code == 401:
                token = await self.get_installation_access_token(installation_id, force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                response = await client.get(url, headers=headers, params=params)

            if response.status_code == 404:
                raise GitHubAppError(f"Repository '{owner}/{repo}' not found.")
            if response.status_code in (403, 429):
                retry_after = response.headers.get("retry-after", "60")
                raise GitHubRateLimitError(f"GitHub API rate limit reached. Retry after {retry_after} seconds.")
            if response.status_code != 200:
                raise GitHubAppError(f"Failed to fetch issues (HTTP {response.status_code}): {response.text}")

            data = response.json()
            results = []
            for iss in data:
                # GitHub issues endpoint includes pull requests unless filtered out
                if iss.get("pull_request"):
                    continue

                user_obj = iss.get("user") or {}
                raw_labels = iss.get("labels", [])
                label_names = [
                    lbl["name"] if isinstance(lbl, dict) else str(lbl)
                    for lbl in raw_labels
                ]

                results.append({
                    "number": iss.get("number"),
                    "title": iss.get("title", ""),
                    "state": iss.get("state", "open"),
                    "author": user_obj.get("login", "Unknown"),
                    "labels": label_names,
                    "createdAt": iss.get("created_at"),
                    "updatedAt": iss.get("updated_at"),
                    "closedAt": iss.get("closed_at"),
                    "htmlUrl": iss.get("html_url"),
                    "commentsCount": iss.get("comments", 0),
                })
            return results

    async def get_repository_live_summary(
        self,
        installation_id: str,
        owner: str,
        repo: str,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Aggregated live summary for the repository.
        Fetches branches, pull requests, and issues concurrently via asyncio.gather,
        resolves active branch, and retrieves the latest commits for that branch.
        """
        branches_task = self.get_repository_branches(installation_id, owner, repo)
        pulls_task = self.get_repository_pull_requests(installation_id, owner, repo, state="all", per_page=15)
        issues_task = self.get_repository_issues(installation_id, owner, repo, state="all", per_page=15)

        branches, pulls, issues = await asyncio.gather(
            branches_task, pulls_task, issues_task
        )

        # Resolve active branch
        branch_names = [b["name"] for b in branches]
        active_branch = branch
        if not active_branch or active_branch not in branch_names:
            if "main" in branch_names:
                active_branch = "main"
            elif "master" in branch_names:
                active_branch = "master"
            elif branch_names:
                active_branch = branch_names[0]
            else:
                active_branch = "main"

        # Update isDefault flag on branches
        for b in branches:
            b["isDefault"] = (b["name"] == active_branch)

        # Fetch commits for the active branch
        commits = await self.get_repository_commits(
            installation_id, owner, repo, branch=active_branch, per_page=15
        )

        open_issues_count = len([i for i in issues if i.get("state") == "open"])
        open_prs_count = len([p for p in pulls if p.get("status") == "open"])
        merged_prs_count = len([p for p in pulls if p.get("status") == "merged"])

        repo_url = f"https://github.com/{owner}/{repo}"

        return {
            "repoOwner": owner,
            "repoName": repo,
            "repoUrl": repo_url,
            "activeBranch": active_branch,
            "branches": branches,
            "commits": commits,
            "pullRequests": pulls,
            "issues": issues,
            "openIssuesCount": open_issues_count,
            "openPRsCount": open_prs_count,
            "mergedPRsCount": merged_prs_count,
        }

    def create_install_state(self, project_id: str, user_id: str) -> str:
        """
        Generates a tamper-proof CSRF state token encoding project_id and user_id.
        Expires in 15 minutes.
        """
        now = int(time.time())
        payload = {
            "pid": project_id,
            "uid": user_id,
            "purpose": "github_install",
            "iat": now,
            "exp": now + 900,  # 15 minutes
        }
        return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

    def verify_install_state(self, state: str, project_id: str, user_id: str) -> bool:
        """
        Verifies that the CSRF state token is valid, unexpired, and matches
        the given project and user.
        """
        try:
            payload = jwt.decode(state, settings.JWT_SECRET, algorithms=["HS256"])
            if payload.get("purpose") != "github_install":
                return False
            if payload.get("pid") != project_id:
                return False
            if payload.get("uid") != user_id:
                return False
            return True
        except Exception as e:
            logger.warning(f"State verification failed: {e}")
            return False

    def get_install_url(self, project_id: str, user_id: str) -> Tuple[str, str]:
        """
        Returns the GitHub App installation URL with an encoded CSRF state token.
        """
        slug = settings.GITHUB_APP_SLUG or "hayyuu-ai"
        state = self.create_install_state(project_id, user_id)
        install_url = f"https://github.com/apps/{slug}/installations/new?state={state}"
        return install_url, state

    async def get_token_for_project(self, project_id: str) -> str:
        """
        Downstream helper for LangGraph agents, MCP servers, and background workers.
        Resolves the valid GitHub Installation Access Token on-demand for a given project ID.
        """
        project = await prisma.project.find_unique(where={"id": project_id})
        if not project:
            raise GitHubAppError(f"Project '{project_id}' not found.")

        if not project.githubInstallationId:
            raise GitHubAppError(
                f"Project '{project.name}' is not connected to a GitHub repository or installation."
            )

        return await self.get_installation_access_token(project.githubInstallationId)


# Singleton service instance
github_service = GitHubService()
