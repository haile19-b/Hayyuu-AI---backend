import logging

from app.infrastructure.graph_store.neo4j import neo4j_graph_store
from app.agents.knowledge_builder.graph import knowledge_builder_graph
from app.agents.knowledge_builder.state import KnowledgeBuilderState

logger = logging.getLogger("uvicorn.error")


class KnowledgeSyncUseCase:
    """Orchestrates purging and synchronizing Neo4j chunk nodes and graph nodes when documents change."""

    async def purge_document_knowledge(self, document_id: str) -> None:
        """
        Delete a document's chunks and detach the Document node in Neo4j.
        """
        logger.info(f"Purging document knowledge for document_id={document_id}")

        # 1. Remove chunks from Neo4j
        cypher_chunks = "MATCH (c:Chunk {document_id: $document_id}) DETACH DELETE c;"
        await neo4j_graph_store.execute_query(cypher_chunks, {"document_id": document_id})
        logger.info(f"Purged Neo4j vector chunks for document_id={document_id}")

        # 2. Detach and delete Document node from Neo4j
        cypher_doc = "MATCH (d:Document {id: $document_id}) DETACH DELETE d;"
        await neo4j_graph_store.execute_query(cypher_doc, {"document_id": document_id})
        logger.info(f"Detached and deleted Document node '{document_id}' from Neo4j.")

    async def sync_document_update(
        self,
        document_id: str,
        project_id: str,
        raw_text: str,
        filename: str = "document",
    ) -> None:
        """
        Synchronize document updates: purges existing states, then runs the unified LangGraph knowledge builder agent.
        """
        logger.info(f"Synchronizing document update for document_id={document_id} under project_id={project_id}")

        # 1. Purge old state to guarantee clean updates
        await self.purge_document_knowledge(document_id)

        # 2. Invoke the unified LangGraph Agent
        state = KnowledgeBuilderState(
            project_id=project_id,
            document_id=document_id,
            raw_text=raw_text,
            filename=filename,
        )
        await knowledge_builder_graph.ainvoke(state)
        logger.info(f"Document update synchronization complete for document_id={document_id}")


# Global singleton instance
knowledge_sync_use_case = KnowledgeSyncUseCase()
