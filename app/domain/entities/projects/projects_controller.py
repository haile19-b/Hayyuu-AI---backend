from typing import Optional
from prisma.enums import ProjectStatus
from app.domain.entities.projects.projects_service import ProjectsService

class ProjectsController:
    @staticmethod
    async def create_project(body, credentials: dict) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await ProjectsService.create_project(
                user_id=user_id,
                name=body.name,
                description=body.description
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_projects(credentials: dict, status: Optional[ProjectStatus] = None) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await ProjectsService.get_projects(user_id=user_id, status=status)
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_by_id(project_id: str, credentials: dict) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await ProjectsService.get_project_by_id(
                project_id=project_id,
                user_id=user_id
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def update_project(project_id: str, body, credentials: dict) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await ProjectsService.update_project(
                project_id=project_id,
                user_id=user_id,
                name=body.name,
                description=body.description,
                status=body.status
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_project(project_id: str, credentials: dict) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await ProjectsService.delete_project(
                project_id=project_id,
                user_id=user_id
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_documents(project_id: str, credentials: dict) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await ProjectsService.get_project_documents(
                project_id=project_id,
                user_id=user_id
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_requirements(project_id: str, credentials: dict) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await ProjectsService.get_project_requirements(
                project_id=project_id,
                user_id=user_id
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_tasks(project_id: str, credentials: dict) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await ProjectsService.get_project_tasks(
                project_id=project_id,
                user_id=user_id
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_suggestions(project_id: str, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.get_project_suggestions(
            project_id=project_id,
            user_id=user_id
        )
      except Exception as e:
        return {"success": False, "error": str(e)}

    @staticmethod
    async def create_requirement(project_id: str, body, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.create_requirement(
            project_id=project_id,
            user_id=user_id,
            title=body.title,
            description=body.description,
            req_type=body.type,
            priority=body.priority
        )
      except Exception as e:
        return {"success": False, "error": str(e)}

    @staticmethod
    async def update_requirement(project_id: str, requirement_id: str, body, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.update_requirement(
            project_id=project_id,
            requirement_id=requirement_id,
            user_id=user_id,
            title=body.title,
            description=body.description,
            req_type=body.type,
            priority=body.priority,
            status=body.status,
            is_conflicted=body.isConflicted
        )
      except Exception as e:
        return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_requirement(project_id: str, requirement_id: str, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.delete_requirement(
            project_id=project_id,
            requirement_id=requirement_id,
            user_id=user_id
        )
      except Exception as e:
        return {"success": False, "error": str(e)}

    @staticmethod
    async def create_task(project_id: str, body, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.create_task(
            project_id=project_id,
            user_id=user_id,
            title=body.title,
            description=body.description,
            requirement_id=body.requirementId,
            priority=body.priority,
            status=body.status
        )
      except Exception as e:
        return {"success": False, "error": str(e)}

    @staticmethod
    async def update_task(project_id: str, task_id: str, body, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.update_task(
            project_id=project_id,
            task_id=task_id,
            user_id=user_id,
            title=body.title,
            description=body.description,
            requirement_id=body.requirementId,
            priority=body.priority,
            status=body.status
        )
      except Exception as e:
        return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_task(project_id: str, task_id: str, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.delete_task(
            project_id=project_id,
            task_id=task_id,
            user_id=user_id
        )
      except Exception as e:
        return {"success": False, "error": str(e)}

    @staticmethod
    async def get_project_conflicts(project_id: str, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.get_project_conflicts(
            project_id=project_id,
            user_id=user_id
        )
      except Exception as e:
        return {"success": False, "error": str(e)}

    @staticmethod
    async def resolve_conflict(project_id: str, conflict_id: str, body, credentials: dict) -> dict:
      try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            return {"success": False, "error": "Unauthorized"}
        return await ProjectsService.resolve_conflict(
            project_id=project_id,
            conflict_id=conflict_id,
            user_id=user_id,
            status=body.status
        )
      except Exception as e:
        return {"success": False, "error": str(e)}
