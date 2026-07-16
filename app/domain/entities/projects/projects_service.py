from typing import Optional
from app.core.database import prisma
from prisma.enums import ProjectStatus

class ProjectsService:
    @staticmethod
    async def create_project(user_id: str, name: str, description: Optional[str] = None) -> dict:
        try:
            project = await prisma.project.create(
                data={
                    "userId": user_id,
                    "name": name,
                    "description": description,
                    "status": ProjectStatus.ACTIVE
                }
            )
            return {
                "success": True,
                "response": project
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_projects(user_id: str, status: Optional[ProjectStatus] = None) -> dict:
        try:
            where_clause = {"userId": user_id}
            if status is not None:
                where_clause["status"] = status
                
            projects = await prisma.project.find_many(
                where=where_clause
            )
            return {
                "success": True,
                "response": projects
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_by_id(project_id: str, user_id: str) -> dict:
        try:
            project = await prisma.project.find_unique(
                where={"id": project_id}
            )
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}
            return {
                "success": True,
                "response": project
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def update_project(
        project_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[ProjectStatus] = None
    ) -> dict:
        try:
            # Verify ownership first
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            update_data = {}
            if name is not None:
                update_data["name"] = name
            if description is not None:
                update_data["description"] = description
            if status is not None:
                update_data["status"] = status

            updated = await prisma.project.update(
                where={"id": project_id},
                data=update_data
            )
            return {
                "success": True,
                "response": updated
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_project(project_id: str, user_id: str) -> dict:
        try:
            # Verify ownership first
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            await prisma.project.delete(where={"id": project_id})
            return {
                "success": True,
                "response": None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_documents(project_id: str, user_id: str) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            documents = await prisma.document.find_many(
                where={"projectId": project_id}
            )
            return {
                "success": True,
                "response": documents
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_requirements(project_id: str, user_id: str) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            requirements = await prisma.requirement.find_many(
                where={"projectId": project_id}
            )
            return {
                "success": True,
                "response": requirements
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_tasks(project_id: str, user_id: str) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            tasks = await prisma.task.find_many(
                where={"projectId": project_id}
            )
            return {
                "success": True,
                "response": tasks
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_suggestions(project_id: str, user_id: str) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            suggestions = await prisma.aisuggestion.find_many(
                where={"projectId": project_id}
            )
            return {
                "success": True,
                "response": suggestions
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
