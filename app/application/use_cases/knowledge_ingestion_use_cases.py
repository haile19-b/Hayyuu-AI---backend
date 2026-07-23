import logging
from typing import Any, Dict, List, Optional

from app.libs.chunker import chunk_document_text
from app.infrastructure.ai.gemini_embedder import gemini_embedder
from app.infrastructure.vector_store.pgvector import pgvector_store

logger = logging.getLogger("uvicorn.error")


class DocumentIngestionUseCase:
    """Use case service for document chunking, embedding, and PGVector ingestion."""

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
        Chunks document text, generates Gemini embeddings, and persists chunks in PGVector.
        """
        if not raw_text.strip():
            logger.warning(f"Empty text received for document_id={document_id}")
            return {
                "document_id": document_id,
                "project_id": project_id,
                "total_chunks": 0,
                "status": "empty",
            }

        # 1. Chunk document text
        chunk_dicts = chunk_document_text(
            raw_text,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        logger.info(f"Generated {len(chunk_dicts)} chunks for document {document_id}")

        # 2. Extract contents and generate embeddings with Gemini
        chunk_texts = [chunk["content"] for chunk in chunk_dicts]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # 3. Attach embeddings and metadata
        for idx, chunk in enumerate(chunk_dicts):
            chunk["embedding"] = embeddings[idx]
            meta = {
                "token_count": chunk["token_count"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
            }
            if extra_metadata:
                meta.update(extra_metadata)
            chunk["metadata"] = meta

        # 4. Upsert into PGVector
        upserted_count = await self.vector_store.upsert_chunks(
            document_id=document_id,
            project_id=project_id,
            chunks=chunk_dicts,
        )

        return {
            "document_id": document_id,
            "project_id": project_id,
            "total_chunks": upserted_count,
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
