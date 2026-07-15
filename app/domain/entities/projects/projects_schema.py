import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field
from prisma.enums import (
    ProjectStatus,
    DocumentStatus,
    RequirementStatus,
    RequirementPriority,
    RequirementType,
    TaskStatus,
    TaskPriority,
    TaskSource,
    TaskApprovalStatus,
    SuggestionStatus
)
from app.domain.entities.auth.auth_schema import ApiResponse

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None

class UpdateProjectRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None

class ProjectResponse(BaseModel):
    id: str
    userId: str
    name: str
    description: Optional[str] = None
    status: ProjectStatus
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    id: str
    projectId: str
    name: str
    filePath: str
    fileType: str
    sizeBytes: int
    status: DocumentStatus
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class RequirementResponse(BaseModel):
    id: str
    projectId: str
    title: str
    description: str
    type: RequirementType
    priority: RequirementPriority
    status: RequirementStatus
    isConflicted: bool
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class TaskResponse(BaseModel):
    id: str
    projectId: str
    requirementId: Optional[str] = None
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    source: TaskSource
    approvalStatus: TaskApprovalStatus
    githubIssueUrl: Optional[str] = None
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class AISuggestionResponse(BaseModel):
    id: str
    projectId: str
    type: str
    content: Any  # Handles dynamic JSON payload
    status: SuggestionStatus
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True
