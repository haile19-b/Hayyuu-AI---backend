from typing import List, Optional
from pydantic import BaseModel, Field


class UserNode(BaseModel):
    id: str = Field(description="Unique ID of the user")
    email: str = Field(description="Email address of the user")


class ProjectNode(BaseModel):
    id: str = Field(description="Unique ID of the project")
    name: str = Field(description="Name of the project")
    description: Optional[str] = Field(None, description="Description of the project")


class DocumentNode(BaseModel):
    id: str = Field(description="Unique ID of the document")
    name: str = Field(description="Name of the document file")
    file_type: str = Field(description="File extension or type (e.g. pdf, docx)")


class RequirementNode(BaseModel):
    id: str = Field(description="Unique ID of the requirement")
    title: str = Field(description="Short title of the requirement")
    description: str = Field(description="Full text description of the requirement")
    type: str = Field(
        "FUNCTIONAL",
        description="Requirement type: 'FUNCTIONAL' or 'NON_FUNCTIONAL'",
    )
    priority: str = Field("P2", description="Priority level: 'P0', 'P1', 'P2'")
    status: str = Field(
        "DRAFT",
        description="Status of the requirement: 'DRAFT', 'SUGGESTION', 'APPROVED', 'REJECTED'",
    )


class TaskNode(BaseModel):
    id: str = Field(description="Unique ID of the task")
    title: str = Field(description="Short title of the task")
    description: str = Field(description="Full description of the work needed")
    status: str = Field(
        "TODO",
        description="Status of the task: 'TODO', 'IN_PROGRESS', 'DONE', 'BLOCKED'",
    )
    priority: str = Field("P2", description="Priority: 'P0', 'P1', 'P2'")


class ConflictNode(BaseModel):
    id: str = Field(description="Unique ID of the conflict relation")
    description: str = Field(description="Description of the conflict between the requirements")
    severity: str = Field("MEDIUM", description="Severity of the conflict: 'HIGH', 'MEDIUM', 'LOW'")
    ai_recommendation: Optional[str] = Field(None, description="AI recommended resolution path")
    requirement_id: str = Field(description="ID of the first requirement")
    conflicting_requirement_id: str = Field(description="ID of the second requirement")


class Relationship(BaseModel):
    source_id: str = Field(description="ID of the source node")
    target_id: str = Field(description="ID of the target node")
    type: str = Field(
        description="Relationship type: 'OWNS', 'HAS_DOCUMENT', 'TRACKS', 'CONTAINS', 'IMPLEMENTS', 'CONFLICTS_WITH'"
    )


class ExtractedKnowledgeGraph(BaseModel):
    """Structured schema for all extracted knowledge entities and relations."""
    users: List[UserNode] = Field(default_factory=list)
    projects: List[ProjectNode] = Field(default_factory=list)
    documents: List[DocumentNode] = Field(default_factory=list)
    requirements: List[RequirementNode] = Field(default_factory=list)
    tasks: List[TaskNode] = Field(default_factory=list)
    conflicts: List[ConflictNode] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
