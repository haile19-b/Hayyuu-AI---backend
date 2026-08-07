import logging
from typing import List, Optional, Dict, Any
from google.genai import types

from app.core.database import prisma
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

logger = logging.getLogger("uvicorn.error")

# =====================================================================
# 1. Tool JSON Schema Definitions using types.FunctionDeclaration
# =====================================================================

list_project_documents_tool = types.FunctionDeclaration(
    name="list_project_documents",
    description="Retrieves metadata of all uploaded documents in the project.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "The active project ID."
            }
        },
        "required": ["project_id"]
    }
)

list_project_requirements_tool = types.FunctionDeclaration(
    name="list_project_requirements",
    description="Retrieves all requirements in the project including type, priority, and status.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "The active project ID."
            }
        },
        "required": ["project_id"]
    }
)

list_project_tasks_tool = types.FunctionDeclaration(
    name="list_project_tasks",
    description="Retrieves all developer tasks in the project including status, priority, and approval flow.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "The active project ID."
            }
        },
        "required": ["project_id"]
    }
)

list_project_conflicts_tool = types.FunctionDeclaration(
    name="list_project_conflicts",
    description="Retrieves all active requirement conflicts in the project along with recommendations.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "The active project ID."
            }
        },
        "required": ["project_id"]
    }
)

list_project_suggestions_tool = types.FunctionDeclaration(
    name="list_project_suggestions",
    description="Retrieves all AI suggested requirements and gaps for the project.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "The active project ID."
            }
        },
        "required": ["project_id"]
    }
)

traverse_project_subgraph_tool = types.FunctionDeclaration(
    name="traverse_project_subgraph",
    description="Retrieves Neo4j relationships and subgraphs for specified entity UUIDs or names.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "The active project ID."
            },
            "entity_ids": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Optional list of UUIDs of entities to traverse."
            },
            "entity_names": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Optional list of names of entities to traverse."
            }
        },
        "required": ["project_id"]
    }
)

get_project_graph_summary_tool = types.FunctionDeclaration(
    name="get_project_graph_summary",
    description="Returns the overall node and edge counts in the project's Neo4j knowledge graph.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "The active project ID."
            }
        },
        "required": ["project_id"]
    }
)

# List of all tools formatted as types.Tool objects or directly passed
search_agent_tools = [
    types.Tool(
        function_declarations=[
            list_project_documents_tool,
            list_project_requirements_tool,
            list_project_tasks_tool,
            list_project_conflicts_tool,
            list_project_suggestions_tool,
            traverse_project_subgraph_tool,
            get_project_graph_summary_tool
        ]
    )
]


# =====================================================================
# 2. Tool Execution Functions
# =====================================================================

async def execute_list_project_documents(project_id: str) -> str:
    """Retrieves metadata of all uploaded documents in the project."""
    logger.info(f"[tool: list_project_documents] Executing for project '{project_id}'")
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


async def execute_list_project_requirements(project_id: str) -> str:
    """Retrieves all requirements in the project including type, priority, and status."""
    logger.info(f"[tool: list_project_requirements] Executing for project '{project_id}'")
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


async def execute_list_project_tasks(project_id: str) -> str:
    """Retrieves all developer tasks in the project including status, priority, and approval flow."""
    logger.info(f"[tool: list_project_tasks] Executing for project '{project_id}'")
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


async def execute_list_project_conflicts(project_id: str) -> str:
    """Retrieves all active requirement conflicts in the project along with recommendations."""
    logger.info(f"[tool: list_project_conflicts] Executing for project '{project_id}'")
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


async def execute_list_project_suggestions(project_id: str) -> str:
    """Retrieves all AI suggested requirements and gaps for the project."""
    logger.info(f"[tool: list_project_suggestions] Executing for project '{project_id}'")
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


async def execute_get_project_graph_summary(project_id: str) -> str:
    """Returns the overall node and edge counts in the project's Neo4j knowledge graph."""
    logger.info(f"[tool: get_project_graph_summary] Executing on project '{project_id}'")
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
