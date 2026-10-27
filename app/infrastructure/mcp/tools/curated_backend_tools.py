import logging
from typing import Optional
from mcp.server import MCPServer
from prisma.enums import TaskStatus, TaskPriority, TaskApprovalStatus, TaskSource

from app.core.database import prisma

logger = logging.getLogger("uvicorn.error")


def register_curated_backend_tools(mcp: MCPServer) -> None:
    """Registers curated FastAPI domain operations as MCP tools."""

    @mcp.tool()
    async def get_project_overview(project_id: str) -> str:
        """
        Retrieves high-level metadata and counts for a project.

        Args:
            project_id: The UUID of the project.
        """
        logger.info(f"[mcp: get_project_overview] Executing for project '{project_id}'")
        try:
            project = await prisma.project.find_unique(
                where={"id": project_id},
                include={
                    "repositories": True,
                    "_count": {
                        "select": {
                            "documents": True,
                            "requirements": True,
                            "tasks": True,
                            "conflicts": True,
                            "suggestions": True,
                        }
                    }
                }
            )
            if not project:
                return f"Error: Project '{project_id}' not found."

            counts = getattr(project, "_count", None) or {}
            repo_info = "None"
            if project.repositories:
                repo_info = ", ".join([r.fullName for r in project.repositories])
            elif project.githubRepoOwner and project.githubRepoName:
                repo_info = f"{project.githubRepoOwner}/{project.githubRepoName}"

            return (
                f"Project Overview: '{project.name}' [ID: {project.id}]\n"
                f"- Description: {project.description or 'No description'}\n"
                f"- Status: {project.status}\n"
                f"- Connected GitHub Repositories: {repo_info}\n"
                f"- Documents Count: {getattr(counts, 'documents', 'N/A')}\n"
                f"- Requirements Count: {getattr(counts, 'requirements', 'N/A')}\n"
                f"- Tasks Count: {getattr(counts, 'tasks', 'N/A')}\n"
                f"- Active Conflicts Count: {getattr(counts, 'conflicts', 'N/A')}\n"
                f"- AI Suggestions Count: {getattr(counts, 'suggestions', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"Error fetching project overview: {e}")
            return f"Error fetching project overview: {str(e)}"

    @mcp.tool()
    async def create_project_task(
        project_id: str,
        title: str,
        description: str,
        priority: str = "P2",
        requirement_id: Optional[str] = None
    ) -> str:
        """
        Creates a new developer task associated with the project and optional requirement.

        Args:
            project_id: The UUID of the project.
            title: The title of the task.
            description: Detailed technical task description and acceptance criteria.
            priority: Priority level (P0, P1, P2). Default is P2.
            requirement_id: Optional UUID of the requirement this task satisfies.
        """
        logger.info(f"[mcp: create_project_task] Creating task '{title}' for project '{project_id}'")
        try:
            # Validate priority
            p_enum = TaskPriority.P2
            if priority.upper() == "P0":
                p_enum = TaskPriority.P0
            elif priority.upper() == "P1":
                p_enum = TaskPriority.P1

            task = await prisma.task.create(
                data={
                    "projectId": project_id,
                    "requirementId": requirement_id,
                    "title": title,
                    "description": description,
                    "priority": p_enum,
                    "status": TaskStatus.TODO,
                    "source": TaskSource.SYSTEM,
                    "approvalStatus": TaskApprovalStatus.APPROVED,
                }
            )
            return f"Successfully created Task '{task.title}' [ID: {task.id}] with status '{task.status}' and priority '{task.priority}'."
        except Exception as e:
            logger.error(f"Error creating project task: {e}")
            return f"Error creating task: {str(e)}"

    @mcp.tool()
    async def update_task_status(task_id: str, status: str) -> str:
        """
        Updates the execution status of a task.

        Args:
            task_id: The UUID of the task.
            status: Target status ('TODO', 'IN_PROGRESS', 'DONE', 'BLOCKED').
        """
        logger.info(f"[mcp: update_task_status] Updating task '{task_id}' to status '{status}'")
        try:
            status_map = {
                "TODO": TaskStatus.TODO,
                "IN_PROGRESS": TaskStatus.IN_PROGRESS,
                "DONE": TaskStatus.DONE,
                "BLOCKED": TaskStatus.BLOCKED,
            }
            normalized = status.upper().strip()
            if normalized not in status_map:
                valid_options = ", ".join(status_map.keys())
                return f"Error: Invalid status '{status}'. Must be one of: {valid_options}"

            task = await prisma.task.update(
                where={"id": task_id},
                data={"status": status_map[normalized]}
            )
            if not task:
                return f"Error: Task '{task_id}' not found."

            return f"Task '{task.title}' [ID: {task.id}] status updated to '{task.status}'."
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            return f"Error updating task status: {str(e)}"

    @mcp.tool()
    async def get_document_parsing_status(document_id: str) -> str:
        """
        Retrieves the indexing and parsing status of an uploaded document.

        Args:
            document_id: The UUID of the document.
        """
        logger.info(f"[mcp: get_document_parsing_status] Checking document '{document_id}'")
        try:
            doc = await prisma.document.find_unique(where={"id": document_id})
            if not doc:
                return f"Error: Document '{document_id}' not found."

            return (
                f"Document '{doc.name}' [ID: {doc.id}]:\n"
                f"- Status: {doc.status}\n"
                f"- File Type: {doc.fileType}\n"
                f"- Size: {doc.sizeBytes} bytes\n"
                f"- Last Updated: {doc.updatedAt.isoformat()}"
            )
        except Exception as e:
            logger.error(f"Error getting document parsing status: {e}")
            return f"Error getting document parsing status: {str(e)}"
