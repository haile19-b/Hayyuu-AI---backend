import logging
from typing import Any, Dict, List, Optional

from app.infrastructure.ai.gemini_embedder import gemini_embedder

logger = logging.getLogger("uvicorn.error")


class DocumentIngestionUseCase:
    """Use case service for orchestrating document ingestion via the unified LangGraph Agent."""

    def __init__(self):
        self.embedder = gemini_embedder

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
        Search for relevant document chunks by generating embedding and querying Neo4j.
        """
        from app.infrastructure.graph_store.neo4j import neo4j_graph_store
        import json
        
        query_vector = self.embedder.embed_text(query_text)
        cypher = """
        CALL db.index.vector.queryNodes('chunk_vector_index', $top_k, $query_vector) YIELD node, score
        WHERE node.project_id = $project_id
          AND ($document_id IS NULL OR node.document_id = $document_id)
          AND score > 0.6
        OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(node)
        OPTIONAL MATCH (node)-[:NEXT]->(next:Chunk)
        RETURN node.id as id,
               node.document_id as document_id,
               node.project_id as project_id,
               node.chunk_index as chunk_index,
               node.content as content,
               prev.content as prev_content,
               next.content as next_content,
               node.metadata as metadata,
               score as similarity
        ORDER BY similarity DESC
        """
        records = await neo4j_graph_store.execute_query(
            cypher,
            {
                "top_k": top_k,
                "query_vector": query_vector,
                "project_id": project_id,
                "document_id": document_id
            }
        )
        
        results = []
        for r in records:
            combined_content = ""
            prev_txt = r.get("prev_content")
            curr_txt = r.get("content") or ""
            next_txt = r.get("next_content")
            
            if prev_txt:
                combined_content += prev_txt.strip() + " \n "
            combined_content += curr_txt.strip()
            if next_txt:
                combined_content += " \n " + next_txt.strip()
                
            raw_meta = r.get("metadata")
            meta = {}
            if raw_meta:
                try:
                    meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                except Exception:
                    meta = {"raw_meta": raw_meta}
                    
            results.append(
                {
                    "document_id": r.get("document_id"),
                    "project_id": r.get("project_id"),
                    "chunk_index": r.get("chunk_index"),
                    "content": combined_content,
                    "metadata": meta,
                    "similarity": float(r.get("similarity") or 0.0),
                }
            )
        return results

    async def remove_document_knowledge(self, document_id: str) -> int:
        """Purge document chunks from Neo4j."""
        from app.infrastructure.graph_store.neo4j import neo4j_graph_store
        cypher = """
        MATCH (c:Chunk {document_id: $doc_id})
        WITH count(c) as cnt, collect(c) as chunks
        UNWIND chunks as c
        DETACH DELETE c
        RETURN cnt
        """
        res = await neo4j_graph_store.execute_query(cypher, {"doc_id": document_id})
        return res[0].get("cnt", 0) if res else 0


# Global singleton instance
document_ingestion_use_case = DocumentIngestionUseCase()
