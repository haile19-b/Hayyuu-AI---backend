from typing import Optional
from app.core.database import prisma
from prisma.enums import (
    ConflictStatus, ProjectStatus, RequirementType, RequirementPriority, RequirementStatus,
    TaskPriority, TaskStatus, TaskSource, TaskApprovalStatus
)

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

    @staticmethod
    async def create_requirement(
        project_id: str,
        user_id: str,
        title: str,
        description: str,
        req_type: RequirementType,
        priority: RequirementPriority
    ) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            requirement = await prisma.requirement.create(
                data={
                    "projectId": project_id,
                    "title": title,
                    "description": description,
                    "type": req_type,
                    "priority": priority,
                    "status": RequirementStatus.DRAFT,
                    "isConflicted": False
                }
            )
            return {
                "success": True,
                "response": requirement
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def update_requirement(
        project_id: str,
        requirement_id: str,
        user_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        req_type: Optional[RequirementType] = None,
        priority: Optional[RequirementPriority] = None,
        status: Optional[RequirementStatus] = None,
        is_conflicted: Optional[bool] = None
    ) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            # Verify requirement exists in project
            requirement = await prisma.requirement.find_unique(where={"id": requirement_id})
            if not requirement or requirement.projectId != project_id:
                return {"success": False, "error": "Requirement not found"}

            update_data = {}
            if title is not None:
                update_data["title"] = title
            if description is not None:
                update_data["description"] = description
            if req_type is not None:
                update_data["type"] = req_type
            if priority is not None:
                update_data["priority"] = priority
            if status is not None:
                update_data["status"] = status
            if is_conflicted is not None:
                update_data["isConflicted"] = is_conflicted

            updated = await prisma.requirement.update(
                where={"id": requirement_id},
                data=update_data
            )
            return {
                "success": True,
                "response": updated
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_requirement(project_id: str, requirement_id: str, user_id: str) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            # Verify requirement exists
            requirement = await prisma.requirement.find_unique(where={"id": requirement_id})
            if not requirement or requirement.projectId != project_id:
                return {"success": False, "error": "Requirement not found"}

            await prisma.requirement.delete(where={"id": requirement_id})
            return {
                "success": True,
                "response": None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def create_task(
        project_id: str,
        user_id: str,
        title: str,
        description: str,
        requirement_id: Optional[str],
        priority: TaskPriority,
        status: TaskStatus
    ) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            task = await prisma.task.create(
                data={
                    "projectId": project_id,
                    "title": title,
                    "description": description,
                    "requirementId": requirement_id,
                    "priority": priority,
                    "status": status,
                    "source": TaskSource.USER,
                    "approvalStatus": TaskApprovalStatus.APPROVED
                }
            )
            return {
                "success": True,
                "response": task
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def update_task(
        project_id: str,
        task_id: str,
        user_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        requirement_id: Optional[str] = None,
        priority: Optional[TaskPriority] = None,
        status: Optional[TaskStatus] = None
    ) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            # Verify task exists
            task = await prisma.task.find_unique(where={"id": task_id})
            if not task or task.projectId != project_id:
                return {"success": False, "error": "Task not found"}

            update_data = {}
            if title is not None:
                update_data["title"] = title
            if description is not None:
                update_data["description"] = description
            if requirement_id is not None:
                update_data["requirementId"] = requirement_id if requirement_id != "none" else None
            if priority is not None:
                update_data["priority"] = priority
            if status is not None:
                update_data["status"] = status

            updated = await prisma.task.update(
                where={"id": task_id},
                data=update_data
            )
            return {
                "success": True,
                "response": updated
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_task(project_id: str, task_id: str, user_id: str) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            # Verify task exists
            task = await prisma.task.find_unique(where={"id": task_id})
            if not task or task.projectId != project_id:
                return {"success": False, "error": "Task not found"}

            await prisma.task.delete(where={"id": task_id})
            return {
                "success": True,
                "response": None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_conflicts(project_id: str, user_id: str) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            conflicts = await prisma.requirementconflict.find_many(
                where={"projectId": project_id}
            )
            return {
                "success": True,
                "response": conflicts
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def resolve_conflict(project_id: str, conflict_id: str, user_id: str, status: ConflictStatus) -> dict:
        try:
            # Verify ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            # Verify conflict exists
            conflict = await prisma.requirementconflict.find_unique(where={"id": conflict_id})
            if not conflict or conflict.projectId != project_id:
                return {"success": False, "error": "Conflict not found"}

            updated = await prisma.requirementconflict.update(
                where={"id": conflict_id},
                data={"status": status}
            )
            return {
                "success": True,
                "response": updated
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
