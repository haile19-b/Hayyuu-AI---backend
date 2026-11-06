"""Project Knowledge Graph Visualization Endpoint (D3.js / vis.js).

Exports Neo4j project knowledge graph subgraphs, nodes, and relationships
formatted for frontend interactive graph visualizers (SRS Module L).
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, Response, status
from app.core.database import prisma
from app.infrastructure.graph_store.neo4j import neo4j_graph_store
from app.middleware.auth_middleware import auth_middleware
from app.domain.entities.auth.auth_schema import ApiResponse

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/projects", tags=["Knowledge Graph Visualizer"])


@router.get("/{project_id}/graph")
async def get_project_graph(
    project_id: str,
    response: Response,
    limit: int = 300,
    credentials = Depends(auth_middleware)
):
    """
    Retrieves the Neo4j knowledge graph for the given project,
    returning nodes and links in standard D3.js / vis.js JSON schema.
    """
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        project = await prisma.project.find_unique(where={"id": project_id})
        if not project:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"success": False, "error": "Project not found"}
        if project.userId != user_id:
            response.status_code = status.HTTP_403_FORBIDDEN
            return {"success": False, "error": "Unauthorized"}

        query = """
        MATCH (n)
        WHERE n.projectId = $project_id OR n.id = $project_id
        OPTIONAL MATCH (n)-[r]->(m)
        WHERE m.projectId = $project_id OR m.id = $project_id
        RETURN n, labels(n) AS n_labels, r, type(r) AS rel_type, m, labels(m) AS m_labels
        LIMIT $limit
        """
        records = await neo4j_graph_store.execute_query(query, {"project_id": project_id, "limit": limit})

        nodes_map: Dict[str, Dict[str, Any]] = {}
        links_list: List[Dict[str, Any]] = []
        seen_links = set()

        for rec in records:
            # Source Node
            n = rec.get("n")
            if n:
                n_id = n.get("id") or n.get("uuid") or str(id(n))
                if n_id not in nodes_map:
                    labels = rec.get("n_labels") or ["Entity"]
                    main_label = labels[0] if labels else "Entity"
                    name = n.get("name") or n.get("title") or n.get("filename") or f"{main_label} ({n_id[:8]})"
                    nodes_map[n_id] = {
                        "id": n_id,
                        "label": main_label,
                        "name": name,
                        "group": main_label,
                        "properties": {k: v for k, v in n.items() if k not in ["embedding"]}
                    }

            # Target Node
            m = rec.get("m")
            if m:
                m_id = m.get("id") or m.get("uuid") or str(id(m))
                if m_id not in nodes_map:
                    labels = rec.get("m_labels") or ["Entity"]
                    main_label = labels[0] if labels else "Entity"
                    name = m.get("name") or m.get("title") or m.get("filename") or f"{main_label} ({m_id[:8]})"
                    nodes_map[m_id] = {
                        "id": m_id,
                        "label": main_label,
                        "name": name,
                        "group": main_label,
                        "properties": {k: v for k, v in m.items() if k not in ["embedding"]}
                    }

            # Relationship / Edge
            r = rec.get("r")
            if r and n and m:
                n_id = n.get("id") or n.get("uuid") or str(id(n))
                m_id = m.get("id") or m.get("uuid") or str(id(m))
                rel_type = rec.get("rel_type") or "RELATED_TO"
                link_key = f"{n_id}->{rel_type}->{m_id}"
                if link_key not in seen_links:
                    seen_links.add(link_key)
                    links_list.append({
                        "id": link_key,
                        "source": n_id,
                        "target": m_id,
                        "type": rel_type,
                        "label": rel_type
                    })

        # Fallback if Neo4j returned no graph nodes yet: populate from relational project entities
        if not nodes_map:
            # Add project root node
            nodes_map[project.id] = {
                "id": project.id,
                "label": "Project",
                "name": project.name,
                "group": "Project",
                "properties": {"description": project.description or "", "status": str(project.status)}
            }
            # Add requirements
            reqs = await prisma.requirement.find_many(where={"projectId": project_id}, take=20)
            for r in reqs:
                nodes_map[r.id] = {
                    "id": r.id,
                    "label": "Requirement",
                    "name": r.title,
                    "group": "Requirement",
                    "properties": {"priority": str(r.priority), "status": str(r.status)}
                }
                links_list.append({
                    "id": f"{project.id}->SPECIFIES->{r.id}",
                    "source": project.id,
                    "target": r.id,
                    "type": "SPECIFIES",
                    "label": "SPECIFIES"
                })

            # Add tasks
            tasks = await prisma.task.find_many(where={"projectId": project_id}, take=20)
            for t in tasks:
                nodes_map[t.id] = {
                    "id": t.id,
                    "label": "Task",
                    "name": t.title,
                    "group": "Task",
                    "properties": {"priority": str(t.priority), "status": str(t.status)}
                }
                target_node = t.requirementId if t.requirementId and t.requirementId in nodes_map else project.id
                links_list.append({
                    "id": f"{t.id}->IMPLEMENTS->{target_node}",
                    "source": t.id,
                    "target": target_node,
                    "type": "IMPLEMENTS",
                    "label": "IMPLEMENTS"
                })

        return {
            "success": True,
            "response": {
                "projectId": project_id,
                "nodes": list(nodes_map.values()),
                "links": links_list,
                "summary": {
                    "nodeCount": len(nodes_map),
                    "linkCount": len(links_list)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error fetching project graph: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}
