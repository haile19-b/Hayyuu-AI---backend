from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Joint Extraction Schema (Requirements, Tasks, and Conflicts)
class ExtractedTask(BaseModel):
    title: str = Field(description="Short milestone title for this task (max 8 words, e.g. 'Implement User Auth API')")
    description: str = Field(default="", description="1 concise sentence defining the scope of this high-level task (under 20 words)")
    priority: str = Field(description="Priority level of the task: 'P0', 'P1', or 'P2'")

class ExtractedRequirement(BaseModel):
    temp_id: str = Field(description="Unique temporary string ID, e.g., 'new_req_1', 'new_req_2'")
    title: str = Field(description="Concise, descriptive title of the requirement (under 10 words)")
    description: str = Field(description="Clear, 1-2 sentence technical explanation of what the system must do")
    type: str = Field(description="Must be either 'FUNCTIONAL' or 'NON_FUNCTIONAL'")
    priority: str = Field(description="Priority level: 'P0', 'P1', or 'P2'")
    tasks: List[ExtractedTask] = Field(default=[], description="1-2 high-level milestone tasks needed to implement this requirement (not detailed subtasks)")

class DetectedConflict(BaseModel):
    conflict_type: str = Field(description="Type of conflict detected: 'EXISTING_VS_NEW' or 'NEW_VS_NEW'")
    existing_requirement_id: Optional[str] = Field(None, description="If EXISTING_VS_NEW, the database UUID of the existing requirement in conflict")
    requirement_temp_id_a: str = Field(description="The temporary ID (e.g., 'new_req_1') of the first newly extracted requirement in conflict")
    requirement_temp_id_b: Optional[str] = Field(None, description="If NEW_VS_NEW, the temporary ID (e.g., 'new_req_2') of the second newly extracted requirement in conflict")
    description: str = Field(description="Concise 1-sentence explanation of the direct contradiction")
    severity: str = Field(description="Severity of the conflict: 'HIGH', 'MEDIUM', or 'LOW'")
    recommendation: str = Field(description="Concise 1-sentence actionable recommendation to resolve this conflict")

class ExtractionResponse(BaseModel):
    requirements: List[ExtractedRequirement]
    conflicts: List[DetectedConflict] = Field(default=[], description="List of detected direct contradictions")


# 2. Gap Analysis & Suggestions Schema
class SuggestedImprovement(BaseModel):
    title: str = Field(description="Short title of the suggested improvement (max 8 words)")
    description: str = Field(description="1 concise sentence explaining what should be added or modified")
    reasoning: str = Field(description="1 concise sentence explaining why this is a gap or key improvement")
    category: str = Field(description="Category of the suggestion: 'security', 'usability', 'performance', 'scalability', or 'other'")

class SuggestionsResponse(BaseModel):
    suggestions: List[SuggestedImprovement]