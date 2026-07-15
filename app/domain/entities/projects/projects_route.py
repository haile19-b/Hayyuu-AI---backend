from typing import List, Optional
from fastapi import APIRouter, Depends, Response, status
from prisma.enums import ProjectStatus
from app.domain.entities.projects.projects_schema import (
    ApiResponse, CreateProjectRequest, UpdateProjectRequest, ProjectResponse,
    DocumentResponse, RequirementResponse, TaskResponse, AISuggestionResponse
)
from app.domain.entities.projects.projects_controller import ProjectsController
from app.middleware.auth_middleware import auth_middleware

router = APIRouter(prefix="/projects", tags=["Projects"])

@router.post("", response_model=ApiResponse[ProjectResponse])
async def create_project(
    body: CreateProjectRequest,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.create_project(body, credentials)
    if not result.get("success"):
        response.status_code = status.HTTP_400_BAD_REQUEST
    else:
        response.status_code = status.HTTP_201_CREATED
    return result

@router.get("", response_model=ApiResponse[List[ProjectResponse]])
async def get_projects(
    response: Response,
    status_filter: Optional[ProjectStatus] = None,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.get_projects(credentials, status=status_filter)
    if not result.get("success"):
        response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.get("/{project_id}", response_model=ApiResponse[ProjectResponse])
async def get_project_by_id(
    project_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.get_project_by_id(project_id, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.put("/{project_id}", response_model=ApiResponse[ProjectResponse])
async def update_project(
    project_id: str,
    body: UpdateProjectRequest,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.update_project(project_id, body, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.delete("/{project_id}", response_model=ApiResponse[None])
async def delete_project(
    project_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.delete_project(project_id, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.get("/{project_id}/documents", response_model=ApiResponse[List[DocumentResponse]])
async def get_project_documents(
    project_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.get_project_documents(project_id, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.get("/{project_id}/requirements", response_model=ApiResponse[List[RequirementResponse]])
async def get_project_requirements(
    project_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.get_project_requirements(project_id, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.get("/{project_id}/tasks", response_model=ApiResponse[List[TaskResponse]])
async def get_project_tasks(
    project_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.get_project_tasks(project_id, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.get("/{project_id}/suggestions", response_model=ApiResponse[List[AISuggestionResponse]])
async def get_project_suggestions(
    project_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await ProjectsController.get_project_suggestions(project_id, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result
