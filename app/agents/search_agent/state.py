from typing import List, Optional, Any
from pydantic import BaseModel, Field
from app.agents.search_agent.schemas import VectorSource, GraphNodeRef

class SearchAgentState(BaseModel):
    """Shared state for the Graph-RAG Search Agent."""
    project_id: str = Field(description="Active project identifier")
    query_text: str = Field(description="The user prompt or query question")
    document_id: Optional[str] = Field(None, description="Optional document identifier to restrict vector search")
    limit: int = Field(5, description="Search limit for vector chunks")
    
    # Retrieved contexts
    chunks: List[VectorSource] = Field(default_factory=list, description="Vector search result chunks")
    graph_nodes: List[GraphNodeRef] = Field(default_factory=list, description="Reference graph nodes involved")
    graph_context: str = Field("", description="Readable string of the Neo4j subgraph relationships")
    postgres_context: str = Field("", description="Accumulated PostgreSQL structured data retrieved via tools")
    
    # Message content history (for Gemini tool calling turns)
    history: List[Any] = Field(default_factory=list, description="Content history objects for LLM context")

    # Output parameters
    answer: str = Field("", description="Synthesized trace-validated response answer")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during agent execution")
    status: str = Field("pending", description="Agent state status: pending, running, completed, failed")
