from typing import List, Optional, Any
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user question or chat prompt.")
    documentId: Optional[str] = Field(None, description="Optional document ID to restrict vector search context.")
    limit: Optional[int] = Field(5, ge=1, le=20, description="Optional similarity result limit.")

class VectorSource(BaseModel):
    id: str = Field(..., description="The unique chunk ID.")
    documentId: str = Field(..., description="The parent document ID.")
    chunkIndex: int = Field(..., description="The chunk's sequential index in the document.")
    content: str = Field(..., description="The text content of the chunk.")
    similarity: float = Field(..., description="The similarity score calculated by cosine distance.")
    metadata: Optional[Any] = Field(None, description="Additional metadata attached to the chunk.")

class GraphNodeRef(BaseModel):
    id: str = Field(..., description="The Neo4j stable entity ID.")
    label: str = Field(..., description="The entity classification label (e.g. Requirement, Task).")
    name: str = Field(..., description="The node's display name or title.")
    properties: Optional[dict] = Field(None, description="Dictionary containing the node's properties.")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The synthesized response from the LLM.")
    sources: List[VectorSource] = Field(default_factory=list, description="Vector sources used in answering.")
    nodes: List[GraphNodeRef] = Field(default_factory=list, description="Graph nodes referenced in answering.")
