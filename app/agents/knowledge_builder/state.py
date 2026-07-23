from typing import List, Optional
from pydantic import BaseModel, Field

from app.agents.knowledge_builder.schema import ExtractedKnowledgeGraph


class KnowledgeBuilderState(BaseModel):
    """LangGraph state representation for the knowledge builder extraction flow."""
    project_id: str = Field(description="Active project identifier")
    document_id: Optional[str] = Field(None, description="Document identifier being parsed")
    raw_text: str = Field(default="", description="Raw text context from documents or database elements")
    extracted_graph: ExtractedKnowledgeGraph = Field(
        default_factory=ExtractedKnowledgeGraph,
        description="Accumulated extracted graph nodes and relations",
    )
    errors: List[str] = Field(default_factory=list, description="Validation/processing errors encountered")
