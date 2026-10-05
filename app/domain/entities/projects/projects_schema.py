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
    SuggestionStatus,
    ConflictSeverity,
    ConflictStatus
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
    documentsCount: int = 0
    requirementsCount: int = 0
    tasksCount: int = 0
    conflictsCount: int = 0

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

class CreateRequirementRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str
    type: RequirementType = RequirementType.FUNCTIONAL
    priority: RequirementPriority = RequirementPriority.P2

class UpdateRequirementRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    type: Optional[RequirementType] = None
    priority: Optional[RequirementPriority] = None
    status: Optional[RequirementStatus] = None
    isConflicted: Optional[bool] = None

class CreateTaskRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str
    requirementId: Optional[str] = None
    priority: TaskPriority = TaskPriority.P2
    status: TaskStatus = TaskStatus.TODO

class UpdateTaskRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    requirementId: Optional[str] = None
    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None

class ConflictResponse(BaseModel):
    id: str
    projectId: str
    requirementId: str
    conflictingRequirementId: str
    severity: ConflictSeverity
    description: str
    aiRecommendation: Optional[str] = None
    status: ConflictStatus
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class ResolveConflictRequest(BaseModel):
    status: ConflictStatus
