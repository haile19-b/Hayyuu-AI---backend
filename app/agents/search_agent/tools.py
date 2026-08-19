import logging
from typing import List, Optional, Any
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

logger = logging.getLogger("uvicorn.error")

async def execute_traverse_project_subgraph(
    project_id: str,
    entity_ids: Optional[List[str]] = None,
    entity_names: Optional[List[str]] = None
) -> Any:
    """Retrieves Neo4j relationships and subgraphs for the specified entity IDs or names."""
    logger.info(f"[tool: traverse_project_subgraph] Executing on project '{project_id}'")
    entity_ids = entity_ids or []
    entity_names = entity_names or []
    try:
        if not entity_ids and not entity_names:
            return {"context": "No entity IDs or names provided for subgraph traversal.", "nodes": [], "rels": []}
            
        rel_query = """
        MATCH (n)-[r]-(m)
        WHERE n.id IN $entity_ids OR n.name IN $entity_names OR m.id IN $entity_ids OR m.name IN $entity_names
        RETURN n, labels(n) as n_labels, type(r) as rel_type, m, labels(m) as m_labels
        LIMIT 100
        """
        rel_records = await neo4j_graph_store.execute_query(
            rel_query,
            {"entity_ids": list(entity_ids), "entity_names": list(entity_names)}
        )
        
        node_query = """
        MATCH (n)
        WHERE n.id IN $entity_ids OR n.name IN $entity_names
        RETURN n, labels(n) as n_labels
        LIMIT 100
        """
        node_records = await neo4j_graph_store.execute_query(
            node_query,
            {"entity_ids": list(entity_ids), "entity_names": list(entity_names)}
        )
        
        # Format Neo4j Subgraph for Prompt Context
        from app.agents.search_agent.nodes import format_neo4j_records
        context_str = format_neo4j_records(rel_records, node_records)
        return {
            "context": context_str,
            "nodes": node_records,
            "rels": rel_records
        }
    except Exception as e:
        logger.error(f"Error traversing subgraph: {e}")
        return {"context": f"Error traversing subgraph: {str(e)}", "nodes": [], "rels": []}
