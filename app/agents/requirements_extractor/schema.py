from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Joint Extraction Schema (Requirements, Tasks, and Conflicts)
class ExtractedTask(BaseModel):
    title: str = Field(description="Title of the developer task to implement this requirement")
    description: str = Field(description="Detailed description of what needs to be done for this task")
    priority: str = Field(description="Priority level of the task: 'P0', 'P1', or 'P2'")

class ExtractedRequirement(BaseModel):
    temp_id: str = Field(description="A unique temporary string ID for this requirement, e.g., 'new_req_1', 'new_req_2'")
    title: str = Field(description="Short, descriptive title of the requirement")
    description: str = Field(description="Detailed explanation of the requirement")
    type: str = Field(description="Must be either 'FUNCTIONAL' or 'NON_FUNCTIONAL'")
    priority: str = Field(description="Priority level: 'P0', 'P1', or 'P2'")
    tasks: List[ExtractedTask] = Field(default=[], description="List of specific developer tasks to implement this requirement")

class DetectedConflict(BaseModel):
    conflict_type: str = Field(description="Type of conflict detected: 'EXISTING_VS_NEW' or 'NEW_VS_NEW'")
    existing_requirement_id: Optional[str] = Field(None, description="If EXISTING_VS_NEW, the database UUID of the existing requirement in conflict")
    requirement_temp_id_a: str = Field(description="The temporary ID (e.g., 'new_req_1') of the first newly extracted requirement in conflict")
    requirement_temp_id_b: Optional[str] = Field(None, description="If NEW_VS_NEW, the temporary ID (e.g., 'new_req_2') of the second newly extracted requirement in conflict")
    description: str = Field(description="Detailed explanation of the contradiction or conflict")
    severity: str = Field(description="Severity of the conflict: 'HIGH', 'MEDIUM', or 'LOW'")
    recommendation: str = Field(description="Actionable AI recommendation to resolve this conflict")

class ExtractionResponse(BaseModel):
    requirements: List[ExtractedRequirement]
    conflicts: List[DetectedConflict] = Field(default=[], description="List of detected conflicts against existing requirements or within the document")


# 2. Gap Analysis & Suggestions Schema
class SuggestedImprovement(BaseModel):
    title: str = Field(description="Title of the suggested improvement or gap resolution")
    description: str = Field(description="Description of what should be added or modified")
    reasoning: str = Field(description="Why this is a gap or key improvement for the project")
    category: str = Field(description="Category of the suggestion: 'security', 'usability', 'performance', 'scalability', or 'other'")

class SuggestionsResponse(BaseModel):
    suggestions: List[SuggestedImprovement]