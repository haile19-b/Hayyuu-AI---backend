from typing import List
from pydantic import BaseModel, Field


class PropertyPair(BaseModel):
    """Key-value representation of dynamic attributes to avoid additionalProperties errors in Gemini Developer API."""
    key: str = Field(description="Name of the attribute")
    value: str = Field(description="Value of the attribute")


class DynamicNode(BaseModel):
    """Dynamic graph node representation."""
    id: str = Field(description="Temporary ID or name used for linking in relationships")
    label: str = Field(description="Entity label, e.g. 'Requirement', 'Task', 'Conflict', 'Component', 'Actor', etc.")
    name: str = Field(description="Descriptive name or title of the entity")
    description: str = Field(description="Main description or details of the entity")
    properties: List[PropertyPair] = Field(default_factory=list, description="Additional custom attributes")


class DynamicRelationship(BaseModel):
    """Dynamic graph relationship representation."""
    source_id: str = Field(description="ID or name of the source node")
    target_id: str = Field(description="ID or name of the target node")
    type: str = Field(description="Relationship type, e.g. 'OWNS', 'HAS_DOCUMENT', 'TRACKS', 'CONTAINS', 'IMPLEMENTS', 'CONFLICTS_WITH'")
    properties: List[PropertyPair] = Field(default_factory=list, description="Additional custom attributes")


class ExtractedKnowledgeGraph(BaseModel):
    """Structured schema for all extracted knowledge entities and relations."""
    nodes: List[DynamicNode] = Field(default_factory=list)
    relationships: List[DynamicRelationship] = Field(default_factory=list)
