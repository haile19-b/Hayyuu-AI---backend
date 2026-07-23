import logging
from typing import Any, Dict, List, Optional

from app.infrastructure.ai.gemini_embedder import gemini_embedder
from app.infrastructure.vector_store.pgvector import pgvector_store

logger = logging.getLogger("uvicorn.error")


class DocumentIngestionUseCase:
    """Use case service for orchestrating document ingestion via the unified LangGraph Agent."""

    def __init__(self):
        self.embedder = gemini_embedder
        self.vector_store = pgvector_store

    async def ingest_document_text(
        self,
        document_id: str,
        project_id: str,
        raw_text: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 100,
    ) -> Dict[str, Any]:
        """
        Ingests document text by running the unified LangGraph knowledge builder agent.
        """
        from app.agents.knowledge_builder.graph import knowledge_builder_graph
        from app.agents.knowledge_builder.state import KnowledgeBuilderState

        filename = (extra_metadata or {}).get("filename", "document")

        logger.info(f"Delegating document text ingestion to unified LangGraph Agent for document {document_id}")
        state = KnowledgeBuilderState(
            project_id=project_id,
            document_id=document_id,
            raw_text=raw_text,
            filename=filename,
        )

        result_state = await knowledge_builder_graph.ainvoke(state)

        if result_state.get("errors"):
            logger.error(f"Unified Agent execution encountered errors: {result_state['errors']}")
            return {
                "document_id": document_id,
                "project_id": project_id,
                "total_chunks": 0,
                "status": "failed",
                "errors": result_state["errors"],
            }

        return {
            "document_id": document_id,
            "project_id": project_id,
            "total_chunks": result_state.get("total_vectors_stored", 0),
            "status": "indexed",
        }

    async def search_document_knowledge(
        self,
        query_text: str,
        project_id: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks by generating embedding for query_text.
        """
        query_vector = self.embedder.embed_text(query_text)
        results = await self.vector_store.search_similar(
            query_vector=query_vector,
            project_id=project_id,
            top_k=top_k,
            document_id=document_id,
        )
        return results

    async def remove_document_knowledge(self, document_id: str) -> int:
        """Purge document chunks from PGVector."""
        return await self.vector_store.delete_document_chunks(document_id)


# Global singleton instance
document_ingestion_use_case = DocumentIngestionUseCase()
