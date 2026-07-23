import logging
from typing import Optional

from app.application.use_cases.knowledge_ingestion_use_cases import document_ingestion_use_case
from app.infrastructure.graph_store.neo4j import neo4j_graph_store
from app.agents.knowledge_builder.graph import knowledge_builder_graph
from app.agents.knowledge_builder.state import KnowledgeBuilderState

logger = logging.getLogger("uvicorn.error")


class KnowledgeSyncUseCase:
    """Orchestrates purging and synchronizing PGVector chunks and Neo4j graph nodes when documents change."""

    async def purge_document_knowledge(self, document_id: str) -> None:
        """
        Delete a document's chunks from PGVector and detach the Document node in Neo4j.
        """
        logger.info(f"Purging document knowledge for document_id={document_id}")

        # 1. Remove chunks from PGVector
        deleted_vector_count = await document_ingestion_use_case.remove_document_knowledge(document_id)
        logger.info(f"Purged {deleted_vector_count} vector chunks for document_id={document_id}")

        # 2. Detach and delete Document node from Neo4j
        cypher = "MATCH (d:Document {id: $document_id}) DETACH DELETE d;"
        await neo4j_graph_store.execute_query(cypher, {"document_id": document_id})
        logger.info(f"Detached and deleted Document node '{document_id}' from Neo4j.")

    async def sync_document_update(
        self,
        document_id: str,
        project_id: str,
        raw_text: str,
        filename: str = "document",
    ) -> None:
        """
        Synchronize document updates: purges existing states, then re-runs chunking, vector ingestion, and graph extraction.
        """
        logger.info(f"Synchronizing document update for document_id={document_id} under project_id={project_id}")

        # 1. Purge old state to guarantee clean updates
        await self.purge_document_knowledge(document_id)

        # 2. Chunk and ingest new vectors to PGVector
        await document_ingestion_use_case.ingest_document_text(
            document_id=document_id,
            project_id=project_id,
            raw_text=raw_text,
            extra_metadata={"filename": filename},
        )

        # 3. Re-extract knowledge graph nodes and relations using LangGraph Agent
        state = KnowledgeBuilderState(
            project_id=project_id,
            document_id=document_id,
            raw_text=raw_text,
        )
        await knowledge_builder_graph.ainvoke(state)
        logger.info(f"Document update synchronization complete for document_id={document_id}")


# Global singleton instance
knowledge_sync_use_case = KnowledgeSyncUseCase()
