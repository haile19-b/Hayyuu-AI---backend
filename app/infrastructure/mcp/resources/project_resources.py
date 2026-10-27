import json
import logging
from typing import Dict, Any
from mcp.server import MCPServer

from app.core.database import prisma
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

logger = logging.getLogger("uvicorn.error")


def register_project_resources(mcp: MCPServer) -> None:
    """Registers MCP Resources on the given MCPServer instance."""

    @mcp.resource("hayyuu://projects/{project_id}/summary", mime_type="application/json")
    async def get_project_summary_resource(project_id: str) -> str:
        """
        Provides a comprehensive JSON snapshot of the project state,
        including document statistics, requirement statuses, task distribution,
        active conflicts, and Neo4j knowledge graph metrics.
        """
        logger.info(f"[mcp: resource] Reading project summary for '{project_id}'")
        try:
            # 1. Fetch project from Prisma
            project = await prisma.project.find_unique(
                where={"id": project_id},
                include={
                    "repositories": True,
                    "documents": True,
                    "requirements": True,
                    "tasks": True,
                    "conflicts": {
                        "where": {"status": "ACTIVE"}
                    },
                    "suggestions": True,
                }
            )

            if not project:
                return json.dumps({"error": f"Project '{project_id}' not found."}, indent=2)

            # 2. Query Neo4j entity counts for this project
            graph_counts: Dict[str, int] = {}
            try:
                query = """
                MATCH (p:Project {id: $project_id})-[r]->(n)
                RETURN labels(n)[0] AS type, count(n) AS count
                """
                records = await neo4j_graph_store.execute_query(query, {"project_id": project_id})
                for r in records:
                    t = r.get("type") or "Unknown"
                    graph_counts[t] = r.get("count", 0)
            except Exception as graph_err:
                logger.warning(f"Failed to query Neo4j metrics for resource: {graph_err}")
                graph_counts["_error"] = str(graph_err)

            # 3. Assemble structured JSON payload
            req_by_status = {}
            for r in project.requirements:
                req_by_status[r.status] = req_by_status.get(r.status, 0) + 1

            tasks_by_status = {}
            for t in project.tasks:
                tasks_by_status[t.status] = tasks_by_status.get(t.status, 0) + 1

            summary_data: Dict[str, Any] = {
                "project_id": project.id,
                "name": project.name,
                "description": project.description,
                "status": project.status,
                "created_at": project.createdAt.isoformat(),
                "updated_at": project.updatedAt.isoformat(),
                "repositories": [
                    {
                        "full_name": r.fullName,
                        "default_branch": r.defaultBranch,
                        "is_active": r.isActive,
                        "html_url": r.htmlUrl,
                    }
                    for r in project.repositories
                ],
                "statistics": {
                    "total_documents": len(project.documents),
                    "total_requirements": len(project.requirements),
                    "requirements_by_status": req_by_status,
                    "total_tasks": len(project.tasks),
                    "tasks_by_status": tasks_by_status,
                    "active_conflicts_count": len(project.conflicts),
                    "suggestions_count": len(project.suggestions),
                    "graph_entities": graph_counts,
                }
            }
            return json.dumps(summary_data, indent=2)
        except Exception as e:
            logger.error(f"Error compiling project summary resource: {e}")
            return json.dumps({"error": str(e)}, indent=2)

    @mcp.resource("hayyuu://projects/{project_id}/schema", mime_type="application/json")
    async def get_project_schema_resource(project_id: str) -> str:
        """
        Exposes the domain ontology and graph schema definition
        to ground LLM reasoning and avoid hallucinations.
        """
        schema_definition = {
            "version": "1.0.0",
            "system": "Hayyuu AI Knowledge Graph & Relational Ontology",
            "project_id": project_id,
            "entities": {
                "Project": {
                    "description": "Root workspace container holding all documents, requirements, and tasks.",
                    "attributes": ["id", "name", "description", "status"]
                },
                "Document": {
                    "description": "Uploaded source files (PDF, DOCX, Markdown, etc.) parsed into vector chunks.",
                    "attributes": ["id", "name", "filePath", "fileType", "sizeBytes", "status"],
                    "statuses": ["PENDING", "INDEXED", "FAILED"]
                },
                "Requirement": {
                    "description": "Functional or non-functional requirement extracted from documents.",
                    "attributes": ["id", "title", "description", "type", "priority", "status", "isConflicted"],
                    "types": ["FUNCTIONAL", "NON_FUNCTIONAL"],
                    "priorities": ["P0", "P1", "P2"],
                    "statuses": ["DRAFT", "SUGGESTION", "APPROVED", "REJECTED"]
                },
                "Task": {
                    "description": "Actionable developer task derived from requirements.",
                    "attributes": ["id", "title", "description", "priority", "status", "source", "approvalStatus"],
                    "statuses": ["TODO", "IN_PROGRESS", "DONE", "BLOCKED"],
                    "priorities": ["P0", "P1", "P2"]
                },
                "RequirementConflict": {
                    "description": "Detected contradiction, overlap, or dependency conflict between requirements.",
                    "attributes": ["id", "requirementId", "conflictingRequirementId", "severity", "status", "description", "aiRecommendation"],
                    "severities": ["HIGH", "MEDIUM", "LOW"]
                }
            },
            "neo4j_graph_relationships": [
                {"from": "Project", "rel": "HAS_DOCUMENT", "to": "Document"},
                {"from": "Document", "rel": "CONTAINS_CHUNK", "to": "Chunk"},
                {"from": "Chunk", "rel": "NEXT", "to": "Chunk", "description": "Sequential chunk flow in source doc"},
                {"from": "Chunk", "rel": "PRIMARY_SOURCE_FOR", "to": "Requirement"},
                {"from": "Chunk", "rel": "MENTIONS", "to": "Entity"},
                {"from": "Requirement", "rel": "HAS_TASK", "to": "Task"},
                {"from": "Requirement", "rel": "CONFLICTS_WITH", "to": "Requirement", "description": "Contradictory specs"}
            ]
        }
        return json.dumps(schema_definition, indent=2)

    @mcp.resource("hayyuu://system/capabilities", mime_type="application/json")
    async def get_system_capabilities_resource() -> str:
        """
        Exposes Hayyuu MCP Server metadata, supported transport layers,
        and high-level capability flags.
        """
        capabilities = {
            "name": "hayyuu-mcp-server",
            "version": "2.0.0",
            "protocol_version": "2024-11-05",
            "supported_transports": ["stdio", "sse", "streamable_http"],
            "features": {
                "tools": True,
                "resources": True,
                "prompts": True,
                "multi_tenant_github": True,
                "curated_fastapi_endpoints": True,
                "vector_rag_neo4j": True,
            },
            "default_sse_endpoint": "/sse",
            "default_messages_endpoint": "/messages"
        }
        return json.dumps(capabilities, indent=2)
