import datetime
from typing import Optional
from pydantic import BaseModel, Field
from prisma.enums import ProjectStatus
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
