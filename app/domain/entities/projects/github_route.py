import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Response, status

from app.core.database import prisma
from app.middleware.auth_middleware import auth_middleware
from app.domain.entities.auth.auth_schema import ApiResponse
from app.domain.entities.projects.projects_schema import (
    GitHubInstallUrlResponse,
    GitHubConnectRequest,
    GitHubSelectRepoRequest,
    GitHubAssignReposRequest,
    GitHubRepositoryItem,
    GitHubStatusResponse,
    ProjectRepositoryResponse,
    GitHubCommitItem,
    GitHubLiveDataResponse,
)
from app.infrastructure.github.github_service import (
    github_service,
    GitHubAppError,
    GitHubConfigError,
    GitHubInstallationNotFoundError,
    GitHubRateLimitError,
)

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/projects/{projectId}/github", tags=["GitHub Integration"])


async def _verify_project_ownership(project_id: str, credentials: dict) -> tuple[Optional[any], Optional[dict]]:
    """Helper to verify authenticated user owns or has access to the project."""
    user_id = credentials.get("id") if credentials else None
    if not user_id:
        return None, {"status": status.HTTP_401_UNAUTHORIZED, "error": "Unauthorized"}

    project = await prisma.project.find_unique(where={"id": project_id})
    if not project:
        return None, {"status": status.HTTP_404_NOT_FOUND, "error": "Project not found"}

    if project.userId != user_id:
        return None, {"status": status.HTTP_403_FORBIDDEN, "error": "You do not have access to this project"}

    return project, None


async def _get_other_claimed_repos(user_id: str, current_project_id: str) -> dict[str, str]:
    """
    Returns a mapping of {fullName: projectName} for repositories claimed
    by other projects owned by the same user to enforce 'One Repo = One Project'.
    """
    other_projects = await prisma.project.find_many(
        where={
            "userId": user_id,
            "NOT": [{"id": current_project_id}],
        },
        include={"repositories": True},
    )
    claimed: dict[str, str] = {}
    for op in other_projects:
        # Check relational ProjectRepository entries
        for r in (op.repositories or []):
            claimed[r.fullName.lower()] = op.name
        # Also check single repo fields for fallback safety
        if op.githubRepoOwner and op.githubRepoName:
            fn = f"{op.githubRepoOwner}/{op.githubRepoName}".lower()
            claimed[fn] = op.name
    return claimed


async def _build_status_response(project, accessible_repos: Optional[List[GitHubRepositoryItem]] = None, suggested_installation_id: Optional[str] = None) -> dict:
    """Builds a consistent GitHubStatusResponse payload."""
    assigned_records = await prisma.projectrepository.find_many(
        where={"projectId": project.id},
        order={"createdAt": "asc"},
    )

    # Backwards compatibility: if project has repoOwner & repoName but no ProjectRepository rows
    if not assigned_records and project.githubRepoOwner and project.githubRepoName:
        fn = f"{project.githubRepoOwner}/{project.githubRepoName}"
        url = project.githubRepoUrl or f"https://github.com/{fn}"
        created = await prisma.projectrepository.create(
            data={
                "projectId": project.id,
                "owner": project.githubRepoOwner,
                "name": project.githubRepoName,
                "fullName": fn,
                "htmlUrl": url,
                "defaultBranch": "main",
                "isActive": True,
            }
        )
        assigned_records = [created]

    assigned_items = [ProjectRepositoryResponse.model_validate(r) for r in assigned_records]
    active_item = next((r for r in assigned_items if r.isActive), None) or (assigned_items[0] if assigned_items else None)

    is_connected = bool(project.githubInstallationId and (project.githubRepoName or assigned_records))

    return {
        "isConnected": is_connected,
        "installationId": project.githubInstallationId,
        "repoOwner": active_item.owner if active_item else project.githubRepoOwner,
        "repoName": active_item.name if active_item else project.githubRepoName,
        "repoUrl": active_item.htmlUrl if active_item else project.githubRepoUrl,
        "connectedAt": project.githubConnectedAt,
        "accessibleRepositories": accessible_repos or [],
        "assignedRepositories": assigned_items,
        "activeRepository": active_item,
        "suggestedInstallationId": suggested_installation_id,
    }


@router.get("/install-url", response_model=ApiResponse[GitHubInstallUrlResponse])
async def get_github_install_url(
    projectId: str,
    response: Response,
    credentials=Depends(auth_middleware),
):
    """
    Returns the GitHub App installation URL containing a tamper-proof CSRF state token
    binding the current project and user.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    try:
        user_id = credentials["id"]
        install_url, state = github_service.get_install_url(project_id=projectId, user_id=user_id)
        return {
            "success": True,
            "response": {
                "installUrl": install_url,
                "state": state,
            },
        }
    except GitHubConfigError as cfg_err:
        logger.error(f"GitHub App configuration error: {cfg_err}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(cfg_err)}
    except Exception as e:
        logger.error(f"Failed to generate GitHub install URL: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": "Failed to generate GitHub App installation URL"}


@router.post("/connect", response_model=ApiResponse[GitHubStatusResponse])
async def connect_github_installation(
    projectId: str,
    body: GitHubConnectRequest,
    response: Response,
    credentials=Depends(auth_middleware),
):
    """
    Connects a GitHub App installation to the project.
    Enforces 'One Repo = One Project': checks if repositories are already claimed by another project.
    If multiple repos are available and none selected, stores the installationId but requires explicit repo selection.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    # If state parameter was provided, verify CSRF state token
    if body.state:
        user_id = credentials["id"]
        if not github_service.verify_install_state(body.state, project_id=projectId, user_id=user_id):
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": "Invalid or expired state verification token."}

    try:
        # Fetch accessible repositories from GitHub for this installation
        repos_data = await github_service.get_installation_repositories(body.installationId)
        user_id = credentials["id"]
        other_claimed = await _get_other_claimed_repos(user_id=user_id, current_project_id=projectId)

        now = datetime.datetime.now(datetime.timezone.utc)
        update_data = {
            "githubInstallationId": body.installationId,
            "githubConnectedAt": now,
        }

        # Handle explicit selected repos if provided in body
        if body.selectedRepos:
            for sr in body.selectedRepos:
                sr_clean = sr.strip().lower()
                if sr_clean in other_claimed:
                    response.status_code = status.HTTP_409_CONFLICT
                    return {
                        "success": False,
                        "error": f"Repository '{sr}' is already assigned to project '{other_claimed[sr_clean]}'.",
                    }

        # If exactly one repo is in the installation, auto-bind it if unclaimed
        if len(repos_data) == 1:
            first_repo = repos_data[0]
            first_fn = first_repo["fullName"].lower()
            if first_fn not in other_claimed:
                update_data["githubRepoOwner"] = first_repo["owner"]
                update_data["githubRepoName"] = first_repo["name"]
                update_data["githubRepoUrl"] = first_repo["htmlUrl"]

                # Upsert into ProjectRepository
                existing_rep = await prisma.projectrepository.find_unique(where={"fullName": first_repo["fullName"]})
                if existing_rep:
                    await prisma.projectrepository.update(
                        where={"id": existing_rep.id},
                        data={"projectId": projectId, "isActive": True},
                    )
                else:
                    await prisma.projectrepository.create(
                        data={
                            "projectId": projectId,
                            "owner": first_repo["owner"],
                            "name": first_repo["name"],
                            "fullName": first_repo["fullName"],
                            "htmlUrl": first_repo["htmlUrl"],
                            "defaultBranch": first_repo.get("defaultBranch", "main"),
                            "isActive": True,
                        }
                    )

        updated_project = await prisma.project.update(
            where={"id": projectId},
            data=update_data,
        )

        # Build accessible repo items with claim metadata
        assigned_records = await prisma.projectrepository.find_many(where={"projectId": projectId})
        assigned_fns = {r.fullName.lower() for r in assigned_records}

        repo_items = []
        for r in repos_data:
            fn_low = r["fullName"].lower()
            repo_items.append(
                GitHubRepositoryItem(
                    **r,
                    isAssignedToThisProject=(fn_low in assigned_fns),
                    isClaimedByOtherProject=(fn_low in other_claimed),
                    claimedByProjectName=other_claimed.get(fn_low),
                )
            )

        status_res = await _build_status_response(updated_project, accessible_repos=repo_items)
        return {"success": True, "response": status_res}

    except GitHubInstallationNotFoundError:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"success": False, "error": "GitHub App installation was not found or was uninstalled."}
    except GitHubRateLimitError as rle:
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {"success": False, "error": str(rle)}
    except GitHubAppError as gae:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "error": str(gae)}
    except Exception as e:
        logger.error(f"Error connecting GitHub installation: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.get("/repositories", response_model=ApiResponse[List[GitHubRepositoryItem]])
async def get_github_repositories(
    projectId: str,
    response: Response,
    credentials=Depends(auth_middleware),
):
    """
    Returns the list of repositories accessible under this project's GitHub App installation.
    Marks repositories already assigned to this project or claimed by other projects.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    installation_id = project.githubInstallationId
    # If not set on this project, check if another project of this user has an installation
    user_id = credentials["id"]
    if not installation_id:
        other_p = await prisma.project.find_first(
            where={"userId": user_id, "githubInstallationId": {"not": None}}
        )
        if other_p:
            installation_id = other_p.githubInstallationId

    if not installation_id:
        try:
            all_installs = await github_service.get_app_installations()
            if all_installs:
                installation_id = str(all_installs[0]["id"])
                await prisma.project.update(
                    where={"id": projectId},
                    data={"githubInstallationId": installation_id},
                )
        except Exception as ex:
            logger.warning(f"Could not check app installations: {ex}")

    if not installation_id:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "error": "Project is not connected to a GitHub App installation."}

    try:
        repos_data = await github_service.get_installation_repositories(installation_id)
        other_claimed = await _get_other_claimed_repos(user_id=user_id, current_project_id=projectId)
        assigned_records = await prisma.projectrepository.find_many(where={"projectId": projectId})
        assigned_fns = {r.fullName.lower() for r in assigned_records}

        repo_items = []
        for r in repos_data:
            fn_low = r["fullName"].lower()
            repo_items.append(
                GitHubRepositoryItem(
                    **r,
                    isAssignedToThisProject=(fn_low in assigned_fns),
                    isClaimedByOtherProject=(fn_low in other_claimed),
                    claimedByProjectName=other_claimed.get(fn_low),
                )
            )

        return {"success": True, "response": repo_items}
    except GitHubInstallationNotFoundError:
        # Installation was revoked on GitHub, clean up project association
        await prisma.projectrepository.delete_many(where={"projectId": projectId})
        await prisma.project.update(
            where={"id": projectId},
            data={
                "githubInstallationId": None,
                "githubRepoOwner": None,
                "githubRepoName": None,
                "githubRepoUrl": None,
                "githubConnectedAt": None,
            },
        )
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"success": False, "error": "GitHub App installation was revoked or removed from GitHub."}
    except GitHubRateLimitError as rle:
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {"success": False, "error": str(rle)}
    except GitHubAppError as gae:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "error": str(gae)}
    except Exception as e:
        logger.error(f"Error fetching repositories: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.post("/select-repo", response_model=ApiResponse[GitHubStatusResponse])
async def select_github_repository(
    projectId: str,
    body: GitHubSelectRepoRequest,
    response: Response,
    credentials=Depends(auth_middleware),
):
    """
    Selects and activates a specific repository for this project.
    Strictly enforces 'One Repo = One Project': rejects if repository is already claimed by another project.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    installation_id = project.githubInstallationId
    user_id = credentials["id"]
    if not installation_id:
        # Fallback 1: check user's installation from another project
        other_p = await prisma.project.find_first(
            where={"userId": user_id, "githubInstallationId": {"not": None}}
        )
        if other_p:
            installation_id = other_p.githubInstallationId
            await prisma.project.update(
                where={"id": projectId},
                data={"githubInstallationId": installation_id},
            )

    if not installation_id:
        # Fallback 2: Check if GitHub App is installed for this specific repository
        try:
            repo_install = await github_service.get_repository_installation(body.owner.strip(), body.repo.strip())
            if repo_install:
                installation_id = str(repo_install["id"])
                await prisma.project.update(
                    where={"id": projectId},
                    data={"githubInstallationId": installation_id},
                )
        except Exception as ex:
            logger.warning(f"Could not query repository installation: {ex}")

    if not installation_id:
        # Fallback 3: Check app installations
        try:
            all_installs = await github_service.get_app_installations()
            if all_installs:
                installation_id = str(all_installs[0]["id"])
                await prisma.project.update(
                    where={"id": projectId},
                    data={"githubInstallationId": installation_id},
                )
        except Exception as ex:
            logger.warning(f"Could not query app installations: {ex}")

    if not installation_id:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {
            "success": False,
            "error": f"Repository '{body.owner}/{body.repo}' is not authorized. Please click 'Connect Repository via GitHub App' to authorize access to this repository on GitHub.",
        }

    full_name = f"{body.owner.strip()}/{body.repo.strip()}"
    full_name_lower = full_name.lower()

    # Verify repository is not claimed by another project
    other_claimed = await _get_other_claimed_repos(user_id=user_id, current_project_id=projectId)
    if full_name_lower in other_claimed:
        claimed_proj = other_claimed[full_name_lower]
        response.status_code = status.HTTP_409_CONFLICT
        return {
            "success": False,
            "error": f"Repository '{full_name}' is already assigned to project '{claimed_proj}'. A repository can only be assigned to one project.",
        }

    try:
        # Verify repository exists in accessible repositories
        repos_data = await github_service.get_installation_repositories(installation_id)
        valid_repo = next((r for r in repos_data if r["fullName"].lower() == full_name_lower), None)
        if not valid_repo:
            # Check if this repository belongs to another installation
            repo_install = await github_service.get_repository_installation(body.owner.strip(), body.repo.strip())
            if repo_install and str(repo_install["id"]) != str(installation_id):
                installation_id = str(repo_install["id"])
                await prisma.project.update(
                    where={"id": projectId},
                    data={"githubInstallationId": installation_id},
                )
                repos_data = await github_service.get_installation_repositories(installation_id)
                valid_repo = next((r for r in repos_data if r["fullName"].lower() == full_name_lower), None)

        if not valid_repo:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "success": False,
                "error": f"Repository '{full_name}' was not found in authorized repositories for this installation. Please click 'Connect Repository via GitHub App' to grant access.",
            }

        now = datetime.datetime.now(datetime.timezone.utc)

        # Deactivate previous active repos in this project
        await prisma.projectrepository.update_many(
            where={"projectId": projectId},
            data={"isActive": False},
        )

        # Upsert repository in project_repositories
        existing_rep = await prisma.projectrepository.find_unique(where={"fullName": valid_repo["fullName"]})
        if existing_rep:
            await prisma.projectrepository.update(
                where={"id": existing_rep.id},
                data={
                    "projectId": projectId,
                    "isActive": True,
                    "defaultBranch": valid_repo.get("defaultBranch", "main"),
                },
            )
        else:
            await prisma.projectrepository.create(
                data={
                    "projectId": projectId,
                    "owner": valid_repo["owner"],
                    "name": valid_repo["name"],
                    "fullName": valid_repo["fullName"],
                    "htmlUrl": valid_repo["htmlUrl"],
                    "defaultBranch": valid_repo.get("defaultBranch", "main"),
                    "isActive": True,
                }
            )

        # Update Project model active fields
        updated_project = await prisma.project.update(
            where={"id": projectId},
            data={
                "githubInstallationId": installation_id,
                "githubRepoOwner": valid_repo["owner"],
                "githubRepoName": valid_repo["name"],
                "githubRepoUrl": valid_repo["htmlUrl"],
                "githubConnectedAt": now,
            },
        )

        repo_items = [GitHubRepositoryItem(**r) for r in repos_data]
        status_res = await _build_status_response(updated_project, accessible_repos=repo_items)

        return {"success": True, "response": status_res}

    except Exception as e:
        logger.error(f"Error selecting repository: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.post("/assign-repos", response_model=ApiResponse[GitHubStatusResponse])
async def assign_github_repositories(
    projectId: str,
    body: GitHubAssignReposRequest,
    response: Response,
    credentials=Depends(auth_middleware),
):
    """
    Assigns one or more repositories to this project's environment.
    Enforces 'One Repo = One Project' across all requested repositories.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    installation_id = project.githubInstallationId
    user_id = credentials["id"]
    if not installation_id:
        other_p = await prisma.project.find_first(
            where={"userId": user_id, "githubInstallationId": {"not": None}}
        )
        if other_p:
            installation_id = other_p.githubInstallationId
            await prisma.project.update(
                where={"id": projectId},
                data={"githubInstallationId": installation_id},
            )

    if not installation_id:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "error": "Project is not connected to a GitHub App installation."}

    other_claimed = await _get_other_claimed_repos(user_id=user_id, current_project_id=projectId)

    for item in body.repositories:
        fn_low = f"{item.owner.strip()}/{item.repo.strip()}".lower()
        if fn_low in other_claimed:
            response.status_code = status.HTTP_409_CONFLICT
            return {
                "success": False,
                "error": f"Repository '{fn_low}' is already assigned to project '{other_claimed[fn_low]}'.",
            }

    try:
        repos_data = await github_service.get_installation_repositories(installation_id)
        repos_by_fn = {r["fullName"].lower(): r for r in repos_data}

        active_target = body.activeRepo.strip().lower() if body.activeRepo else None

        for item in body.repositories:
            fn_low = f"{item.owner.strip()}/{item.repo.strip()}".lower()
            valid_repo = repos_by_fn.get(fn_low)
            if not valid_repo:
                continue

            is_active = (active_target == fn_low or active_target == valid_repo["name"].lower())

            existing_rep = await prisma.projectrepository.find_unique(where={"fullName": valid_repo["fullName"]})
            if existing_rep:
                await prisma.projectrepository.update(
                    where={"id": existing_rep.id},
                    data={
                        "projectId": projectId,
                        "isActive": is_active,
                        "defaultBranch": valid_repo.get("defaultBranch", "main"),
                    },
                )
            else:
                await prisma.projectrepository.create(
                    data={
                        "projectId": projectId,
                        "owner": valid_repo["owner"],
                        "name": valid_repo["name"],
                        "fullName": valid_repo["fullName"],
                        "htmlUrl": valid_repo["htmlUrl"],
                        "defaultBranch": valid_repo.get("defaultBranch", "main"),
                        "isActive": is_active,
                    }
                )

        now = datetime.datetime.now(datetime.timezone.utc)
        updated_project = await prisma.project.update(
            where={"id": projectId},
            data={"githubConnectedAt": now},
        )

        repo_items = [GitHubRepositoryItem(**r) for r in repos_data]
        status_res = await _build_status_response(updated_project, accessible_repos=repo_items)
        return {"success": True, "response": status_res}

    except Exception as e:
        logger.error(f"Error assigning repositories: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.get("/status", response_model=ApiResponse[GitHubStatusResponse])
async def get_github_status(
    projectId: str,
    response: Response,
    credentials=Depends(auth_middleware),
):
    """
    Returns current GitHub integration connection status, assigned repositories,
    active repository, and accessible repositories from GitHub.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    user_id = credentials["id"]

    # Check if project has an installation
    if not project.githubInstallationId:
        # Check if user has an existing installation in another project
        other_project = await prisma.project.find_first(
            where={
                "userId": user_id,
                "githubInstallationId": {"not": None},
                "NOT": [{"id": projectId}],
            }
        )

        accessible_repos: List[GitHubRepositoryItem] = []
        suggested_id = other_project.githubInstallationId if other_project else None

        if not suggested_id:
            try:
                all_installs = await github_service.get_app_installations()
                if all_installs:
                    suggested_id = str(all_installs[0]["id"])
            except Exception as ex:
                logger.warning(f"Could not check app installations in status: {ex}")

        if suggested_id:
            try:
                repos_data = await github_service.get_installation_repositories(suggested_id)
                other_claimed = await _get_other_claimed_repos(user_id=user_id, current_project_id=projectId)
                for r in repos_data:
                    fn_low = r["fullName"].lower()
                    accessible_repos.append(
                        GitHubRepositoryItem(
                            **r,
                            isAssignedToThisProject=False,
                            isClaimedByOtherProject=(fn_low in other_claimed),
                            claimedByProjectName=other_claimed.get(fn_low),
                        )
                    )
            except Exception as ex:
                logger.warning(f"Could not load accessible repos for suggested installation: {ex}")

        status_res = await _build_status_response(
            project,
            accessible_repos=accessible_repos,
            suggested_installation_id=suggested_id,
        )
        return {"success": True, "response": status_res}

    # Fetch accessible repos to check if installation is active
    accessible_repos = []
    try:
        repos_data = await github_service.get_installation_repositories(project.githubInstallationId)
        other_claimed = await _get_other_claimed_repos(user_id=user_id, current_project_id=projectId)
        assigned_records = await prisma.projectrepository.find_many(where={"projectId": projectId})
        assigned_fns = {r.fullName.lower() for r in assigned_records}

        for r in repos_data:
            fn_low = r["fullName"].lower()
            accessible_repos.append(
                GitHubRepositoryItem(
                    **r,
                    isAssignedToThisProject=(fn_low in assigned_fns),
                    isClaimedByOtherProject=(fn_low in other_claimed),
                    claimedByProjectName=other_claimed.get(fn_low),
                )
            )
    except GitHubInstallationNotFoundError:
        # Installation was deleted on GitHub; unlink automatically
        await prisma.projectrepository.delete_many(where={"projectId": projectId})
        await prisma.project.update(
            where={"id": projectId},
            data={
                "githubInstallationId": None,
                "githubRepoOwner": None,
                "githubRepoName": None,
                "githubRepoUrl": None,
                "githubConnectedAt": None,
            },
        )
        status_res = await _build_status_response(project, accessible_repos=[])
        return {"success": True, "response": status_res}
    except Exception as ex:
        logger.warning(f"Could not refresh accessible repos for project {projectId}: {ex}")

    status_res = await _build_status_response(project, accessible_repos=accessible_repos)
    return {"success": True, "response": status_res}


@router.get("/live-data", response_model=ApiResponse[GitHubLiveDataResponse])
async def get_github_live_data(
    projectId: str,
    response: Response,
    repo: Optional[str] = Query(None),
    branch: Optional[str] = Query(None),
    credentials=Depends(auth_middleware),
):
    """
    Returns real-time repository data (branches, commits, PRs, issues, counts)
    for the specified or active repository assigned to this project.
    Strictly prevents access to repositories not assigned to this project.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    if not project.githubInstallationId:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "error": "Project is not connected to a GitHub App installation."}

    # Resolve target repository from project's assigned repositories
    assigned_repos = await prisma.projectrepository.find_many(where={"projectId": projectId})
    target_owner: Optional[str] = None
    target_repo: Optional[str] = None

    if repo:
        repo_clean = repo.strip().lower()
        matched = next(
            (r for r in assigned_repos if r.name.lower() == repo_clean or r.fullName.lower() == repo_clean),
            None,
        )
        if not matched:
            # Check legacy single repo fields on Project
            if project.githubRepoName and (
                project.githubRepoName.lower() == repo_clean
                or f"{project.githubRepoOwner}/{project.githubRepoName}".lower() == repo_clean
            ):
                target_owner = project.githubRepoOwner
                target_repo = project.githubRepoName
            else:
                response.status_code = status.HTTP_403_FORBIDDEN
                return {
                    "success": False,
                    "error": f"Repository '{repo}' is not assigned to this project.",
                }
        else:
            target_owner = matched.owner
            target_repo = matched.name
            if not matched.isActive:
                await prisma.projectrepository.update_many(
                    where={"projectId": projectId},
                    data={"isActive": False},
                )
                await prisma.projectrepository.update(
                    where={"id": matched.id},
                    data={"isActive": True},
                )
                await prisma.project.update(
                    where={"id": projectId},
                    data={
                        "githubRepoOwner": matched.owner,
                        "githubRepoName": matched.name,
                        "githubRepoUrl": matched.htmlUrl or f"https://github.com/{matched.fullName}",
                    },
                )
    else:
        # Use active assigned repository or fallback to Project active fields
        active_rec = next((r for r in assigned_repos if r.isActive), None)
        if active_rec:
            target_owner = active_rec.owner
            target_repo = active_rec.name
        elif project.githubRepoOwner and project.githubRepoName:
            target_owner = project.githubRepoOwner
            target_repo = project.githubRepoName
        elif assigned_repos:
            target_owner = assigned_repos[0].owner
            target_repo = assigned_repos[0].name
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": "No repository is linked to this project."}

    try:
        live_summary = await github_service.get_repository_live_summary(
            installation_id=project.githubInstallationId,
            owner=target_owner,
            repo=target_repo,
            branch=branch,
        )
        return {"success": True, "response": live_summary}
    except GitHubRateLimitError as rle:
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {"success": False, "error": str(rle)}
    except GitHubAppError as gae:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "error": str(gae)}
    except Exception as e:
        logger.error(f"Error fetching live repository data: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.get("/commits", response_model=ApiResponse[List[GitHubCommitItem]])
async def get_github_commits(
    projectId: str,
    response: Response,
    repo: Optional[str] = Query(None),
    branch: Optional[str] = Query(None),
    per_page: int = Query(15, ge=1, le=100),
    credentials=Depends(auth_middleware),
):
    """
    Returns real-time commits for a specific branch and repository.
    Strictly prevents querying repositories not assigned to this project.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    if not project.githubInstallationId:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "error": "Project is not connected to a GitHub App installation."}

    assigned_repos = await prisma.projectrepository.find_many(where={"projectId": projectId})
    target_owner: Optional[str] = None
    target_repo: Optional[str] = None

    if repo:
        repo_clean = repo.strip().lower()
        matched = next(
            (r for r in assigned_repos if r.name.lower() == repo_clean or r.fullName.lower() == repo_clean),
            None,
        )
        if not matched:
            if project.githubRepoName and (
                project.githubRepoName.lower() == repo_clean
                or f"{project.githubRepoOwner}/{project.githubRepoName}".lower() == repo_clean
            ):
                target_owner = project.githubRepoOwner
                target_repo = project.githubRepoName
            else:
                response.status_code = status.HTTP_403_FORBIDDEN
                return {
                    "success": False,
                    "error": f"Repository '{repo}' is not assigned to this project.",
                }
        else:
            target_owner = matched.owner
            target_repo = matched.name
    else:
        active_rec = next((r for r in assigned_repos if r.isActive), None)
        if active_rec:
            target_owner = active_rec.owner
            target_repo = active_rec.name
        elif project.githubRepoOwner and project.githubRepoName:
            target_owner = project.githubRepoOwner
            target_repo = project.githubRepoName
        elif assigned_repos:
            target_owner = assigned_repos[0].owner
            target_repo = assigned_repos[0].name
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": "No repository is linked to this project."}

    try:
        commits = await github_service.get_repository_commits(
            installation_id=project.githubInstallationId,
            owner=target_owner,
            repo=target_repo,
            branch=branch,
            per_page=per_page,
        )
        return {"success": True, "response": commits}
    except GitHubRateLimitError as rle:
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {"success": False, "error": str(rle)}
    except GitHubAppError as gae:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "error": str(gae)}
    except Exception as e:
        logger.error(f"Error fetching commits: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.delete("/disconnect", response_model=ApiResponse[dict])
async def disconnect_github(
    projectId: str,
    response: Response,
    repo: Optional[str] = Query(None),
    credentials=Depends(auth_middleware),
):
    """
    Unlinks a repository or entire GitHub integration from the project record.
    If 'repo' query parameter is specified, unlinks only that repository.
    Other projects and the underlying GitHub installation remain completely intact.
    """
    project, err = await _verify_project_ownership(projectId, credentials)
    if err:
        response.status_code = err["status"]
        return {"success": False, "error": err["error"]}

    try:
        if repo:
            repo_clean = repo.strip().lower()
            # Delete repository binding
            assigned_repos = await prisma.projectrepository.find_many(where={"projectId": projectId})
            target = next(
                (r for r in assigned_repos if r.name.lower() == repo_clean or r.fullName.lower() == repo_clean),
                None,
            )
            if target:
                await prisma.projectrepository.delete(where={"id": target.id})

            # Check remaining assigned repos
            remaining = await prisma.projectrepository.find_many(where={"projectId": projectId})
            if remaining:
                next_rep = remaining[0]
                await prisma.projectrepository.update(
                    where={"id": next_rep.id},
                    data={"isActive": True},
                )
                await prisma.project.update(
                    where={"id": projectId},
                    data={
                        "githubRepoOwner": next_rep.owner,
                        "githubRepoName": next_rep.name,
                        "githubRepoUrl": next_rep.htmlUrl,
                    },
                )
            else:
                await prisma.project.update(
                    where={"id": projectId},
                    data={
                        "githubRepoOwner": None,
                        "githubRepoName": None,
                        "githubRepoUrl": None,
                    },
                )

            return {
                "success": True,
                "message": f"Repository '{repo}' successfully disconnected from project.",
                "response": {"disconnected": True, "remainingCount": len(remaining)},
            }
        else:
            # Unlink all repositories and installation from this project
            await prisma.projectrepository.delete_many(where={"projectId": projectId})
            await prisma.project.update(
                where={"id": projectId},
                data={
                    "githubInstallationId": None,
                    "githubRepoOwner": None,
                    "githubRepoName": None,
                    "githubRepoUrl": None,
                    "githubConnectedAt": None,
                },
            )
            return {
                "success": True,
                "message": "GitHub repository successfully disconnected.",
                "response": {"disconnected": True},
            }
    except Exception as e:
        logger.error(f"Error disconnecting GitHub: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}
