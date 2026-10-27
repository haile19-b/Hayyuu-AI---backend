import logging
from typing import List, Optional
from mcp.server import MCPServer

from app.core.database import prisma
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

logger = logging.getLogger("uvicorn.error")


def register_db_graph_tools(mcp: MCPServer) -> None:
    """Registers core database and Neo4j graph tools on the given MCPServer instance."""

    @mcp.tool()
    async def list_project_documents(project_id: str) -> str:
        """
        Retrieves metadata of all uploaded documents in the project.

        Args:
            project_id: The UUID of the active project.
        """
        logger.info(f"[mcp: list_project_documents] Executing for project '{project_id}'")
        try:
            docs = await prisma.document.find_many(where={"projectId": project_id})
            if not docs:
                return "No documents found in this project."
            return "\n".join([
                f"- Document '{d.name}' [ID: {d.id}]: Status={d.status}, Type={d.fileType}, Size={d.sizeBytes} bytes"
                for d in docs
            ])
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return f"Error listing documents: {str(e)}"

    @mcp.tool()
    async def list_project_requirements(project_id: str) -> str:
        """
        Retrieves all requirements in the project including type, priority, and status.

        Args:
            project_id: The UUID of the active project.
        """
        logger.info(f"[mcp: list_project_requirements] Executing for project '{project_id}'")
        try:
            reqs = await prisma.requirement.find_many(where={"projectId": project_id})
            if not reqs:
                return "No requirements found in this project."
            return "\n".join([
                f"- Requirement '{r.title}' [ID: {r.id}]: Type={r.type}, Priority={r.priority}, Status={r.status}, Conflicted={r.isConflicted}"
                for r in reqs
            ])
        except Exception as e:
            logger.error(f"Error listing requirements: {e}")
            return f"Error listing requirements: {str(e)}"

    @mcp.tool()
    async def list_project_tasks(project_id: str) -> str:
        """
        Retrieves all developer tasks in the project including status, priority, and approval flow.

        Args:
            project_id: The UUID of the active project.
        """
        logger.info(f"[mcp: list_project_tasks] Executing for project '{project_id}'")
        try:
            tasks = await prisma.task.find_many(where={"projectId": project_id})
            if not tasks:
                return "No tasks found in this project."
            return "\n".join([
                f"- Task '{t.title}' [ID: {t.id}]: Status={t.status}, Priority={t.priority}, Approval={t.approvalStatus}, Source={t.source}"
                for t in tasks
            ])
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return f"Error listing tasks: {str(e)}"

    @mcp.tool()
    async def list_project_conflicts(project_id: str) -> str:
        """
        Retrieves all active requirement conflicts in the project along with recommendations.

        Args:
            project_id: The UUID of the active project.
        """
        logger.info(f"[mcp: list_project_conflicts] Executing for project '{project_id}'")
        try:
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
        except Exception as e:
            logger.error(f"Error listing conflicts: {e}")
            return f"Error listing conflicts: {str(e)}"

    @mcp.tool()
    async def list_project_suggestions(project_id: str) -> str:
        """
        Retrieves all AI suggested requirements and gaps for the project.

        Args:
            project_id: The UUID of the active project.
        """
        logger.info(f"[mcp: list_project_suggestions] Executing for project '{project_id}'")
        try:
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
        except Exception as e:
            logger.error(f"Error listing suggestions: {e}")
            return f"Error listing suggestions: {str(e)}"

    @mcp.tool()
    async def get_project_graph_summary(project_id: str) -> str:
        """
        Returns the overall node and edge counts in the project's Neo4j knowledge graph.

        Args:
            project_id: The UUID of the active project.
        """
        logger.info(f"[mcp: get_project_graph_summary] Executing on project '{project_id}'")
        try:
            query = """
            MATCH (p:Project {id: $project_id})-[r]->(n)
            RETURN labels(n)[0] AS type, count(n) AS count
            """
            records = await neo4j_graph_store.execute_query(query, {"project_id": project_id})
            if not records:
                return "No graph metadata exists in Neo4j for this project."
            summary = [f"- {r.get('type', 'Entity')}: {r.get('count', 0)} nodes" for r in records]
            return "Project Graph Summary:\n" + "\n".join(summary)
        except Exception as e:
            logger.error(f"Error getting graph summary: {e}")
            return f"Error getting graph summary: {str(e)}"

    @mcp.tool()
    async def traverse_project_subgraph(
        project_id: str,
        entity_ids: Optional[List[str]] = None,
        entity_names: Optional[List[str]] = None
    ) -> str:
        """
        Retrieves Neo4j relationships and subgraphs for specified entity UUIDs or names.

        Args:
            project_id: The UUID of the active project.
            entity_ids: Optional list of UUIDs of entities to traverse.
            entity_names: Optional list of names of entities to traverse.
        """
        logger.info(f"[mcp: traverse_project_subgraph] Executing on project '{project_id}'")
        entity_ids = entity_ids or []
        entity_names = entity_names or []
        try:
            if not entity_ids and not entity_names:
                return "No entity IDs or names provided for subgraph traversal."

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

            lines = []
            seen_nodes = set()

            lines.append("Nodes:")
            for record in node_records:
                n = record.get("n") or {}
                n_id = n.get("id", "Unknown")
                if n_id in seen_nodes:
                    continue
                seen_nodes.add(n_id)
                n_labels = record.get("n_labels") or ["Entity"]
                n_label = n_labels[0] if n_labels else "Entity"
                n_name = n.get("name") or n.get("title") or "Unnamed"
                node_str = f"- {n_label}: '{n_name}' [ID: {n_id}]"
                lines.append(node_str)

            if rel_records:
                lines.append("\nRelationships:")
                seen_rels = set()
                for record in rel_records:
                    n = record.get("n") or {}
                    m = record.get("m") or {}
                    n_labels = record.get("n_labels") or ["Entity"]
                    m_labels = record.get("m_labels") or ["Entity"]
                    rel_type = record.get("rel_type") or "RELATED_TO"
                    n_id = n.get("id", "")
                    m_id = m.get("id", "")
                    rel_key = f"{n_id}-{rel_type}-{m_id}"
                    if rel_key in seen_rels:
                        continue
                    seen_rels.add(rel_key)
                    n_label = n_labels[0]
                    m_label = m_labels[0]
                    n_name = n.get("name") or n.get("title") or "Unnamed"
                    m_name = m.get("name") or m.get("title") or "Unnamed"
                    rel_str = f"- ({n_label}: '{n_name}' [ID: {n_id}]) -[{rel_type}]-> ({m_label}: '{m_name}' [ID: {m_id}])"
                    lines.append(rel_str)

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error traversing subgraph: {e}")
            return f"Error traversing subgraph: {str(e)}"


