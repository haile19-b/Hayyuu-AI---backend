import logging
from typing import List, Optional, Dict, Any

from app.core.database import prisma
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

logger = logging.getLogger("uvicorn.error")


async def relational_db_tool(
    project_id: str,
    action: str,
    params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Dedicated tool for PostgreSQL database access.
    
    Supported Actions:
      - 'list_documents': Retrieves all uploaded document metadata for the project.
      - 'list_requirements': Retrieves all project requirements including type, priority, and status.
      - 'list_tasks': Retrieves all task details including priority, status, and approval flow.
      - 'list_conflicts': Retrieves active requirement conflicts along with recommendations.
      - 'list_suggestions': Retrieves AI suggested requirements/gaps.
    """
    params = params or {}
    logger.info(f"[relational_db_tool] Executing action '{action}' on project '{project_id}'")
    
    try:
        if action == "list_documents":
            docs = await prisma.document.find_many(where={"projectId": project_id})
            if not docs:
                return "No documents found in this project."
            return "\n".join([
                f"- Document '{d.name}' [ID: {d.id}]: Status={d.status}, Type={d.fileType}, Size={d.sizeBytes} bytes"
                for d in docs
            ])
            
        elif action == "list_requirements":
            reqs = await prisma.requirement.find_many(where={"projectId": project_id})
            if not reqs:
                return "No requirements found in this project."
            return "\n".join([
                f"- Requirement '{r.title}' [ID: {r.id}]: Type={r.type}, Priority={r.priority}, Status={r.status}, Conflicted={r.isConflicted}"
                for r in reqs
            ])
            
        elif action == "list_tasks":
            tasks = await prisma.task.find_many(where={"projectId": project_id})
            if not tasks:
                return "No tasks found in this project."
            return "\n".join([
                f"- Task '{t.title}' [ID: {t.id}]: Status={t.status}, Priority={t.priority}, Approval={t.approvalStatus}, Source={t.source}"
                for t in tasks
            ])
            
        elif action == "list_conflicts":
            conflicts = await prisma.requirementconflict.find_many(
                where={"projectId": project_id, "status": "ACTIVE"},
                include={"requirement": True, "conflictingRequirement": True}
            )
            if not conflicts:
                return "No active conflicts found in this project."
            return "\n".join([
                f"- Conflict [ID: {c.id}] (Severity: {c.severity}): Description={c.description}, Recommendation={c.aiRecommendation or 'None'}, "
                f"Requirements involved: Requirement A ID={c.requirementId} ({c.requirement.title if c.requirement else 'Unknown'}), "
                f"Requirement B ID={c.conflictingRequirementId} ({c.conflictingRequirement.title if c.conflictingRequirement else 'Unknown'})"
                for c in conflicts
            ])
            
        elif action == "list_suggestions":
            suggestions = await prisma.aisuggestion.find_many(where={"projectId": project_id})
            if not suggestions:
                return "No AI suggestions found in this project."
            lines = []
            for s in suggestions:
                content = s.content
                title = content.get("title") if isinstance(content, dict) else "Unnamed Suggestion"
                desc = content.get("description") if isinstance(content, dict) else str(content)
                lines.append(f"- AI Suggestion '{title}' [ID: {s.id}] (Type: {s.type}): Status={s.status}, Description={desc}")
            return "\n".join(lines)
            
        else:
            return f"Unsupported relational database action: '{action}'"
            
    except Exception as e:
        logger.error(f"[relational_db_tool] Error executing action '{action}': {e}")
        return f"Error executing relational database action '{action}': {str(e)}"


async def graph_db_tool(
    project_id: str,
    action: str,
    params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Dedicated tool for Neo4j graph database access.
    
    Supported Actions:
      - 'traverse_subgraph': Retrieves relationship paths for a list of entity IDs/names.
                            Required param keys: 'entity_ids' (List[str]) or 'entity_names' (List[str]).
      - 'get_project_graph_summary': Returns overall node/edge counts for this project.
    """
    params = params or {}
    logger.info(f"[graph_db_tool] Executing action '{action}' on project '{project_id}'")
    
    try:
        if action == "traverse_subgraph":
            entity_ids = params.get("entity_ids") or []
            entity_names = params.get("entity_names") or []
            
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
            
        elif action == "get_project_graph_summary":
            query = """
            MATCH (p:Project {id: $project_id})-[r]->(n)
            RETURN labels(n)[0] AS type, count(n) AS count
            """
            records = await neo4j_graph_store.execute_query(query, {"project_id": project_id})
            if not records:
                return "No graph metadata exists in Neo4j for this project."
            summary = [f"- {r.get('type', 'Entity')}: {r.get('count', 0)} nodes" for r in records]
            return "Project Graph Summary:\n" + "\n".join(summary)
            
        else:
            return f"Unsupported graph database action: '{action}'"
            
    except Exception as e:
        logger.error(f"[graph_db_tool] Error executing action '{action}': {e}")
        return f"Error executing graph database action '{action}': {str(e)}"
