from typing import List, Optional
from pydantic import BaseModel, Field

from app.agents.knowledge_builder.schema import ExtractedKnowledgeGraph


class ChunkData(BaseModel):
    """Intermediate data structure representing text chunks and vector embeddings."""
    chunk_index: int
    content: str
    token_count: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None


class KnowledgeBuilderState(BaseModel):
    """LangGraph state representation for the end-to-end knowledge builder agent."""
    project_id: str = Field(description="Active project identifier")
    document_id: Optional[str] = Field(None, description="Document identifier (optional)")
    filename: str = Field("document", description="Filename reference (optional)")
    raw_text: str = Field("", description="Raw text context from documents or database elements")
    
    # Conditional file input fields (multimodal Gemini Files API)
    gemini_file_uri: Optional[str] = Field(None, description="Gemini Files API URI reference")
    gemini_file_mime_type: Optional[str] = Field(None, description="Gemini Files API Mime type")

    # Intermediate flow variables
    chunks: List[ChunkData] = Field(default_factory=list, description="List of preprocessed text chunks")
    extracted_graph: ExtractedKnowledgeGraph = Field(
        default_factory=ExtractedKnowledgeGraph,
        description="Accumulated extracted graph nodes and relations",
    )

    # Execution metrics and results
    total_vectors_stored: int = Field(0, description="Count of chunks stored in PGVector")
    total_nodes_stored: int = Field(0, description="Count of entities/relations written to Neo4j")
    errors: List[str] = Field(default_factory=list, description="Validation/processing errors encountered")
    status: str = Field("pending", description="Pipeline status: pending, running, completed, failed")
