import logging
from typing import Any, Dict, List, Optional
from app.agents.knowledge_builder.schema import ExtractedKnowledgeGraph
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

logger = logging.getLogger("uvicorn.error")


async def write_knowledge_graph_to_neo4j(
    project_id: str,
    graph: ExtractedKnowledgeGraph,
) -> None:
    """Idempotently batch write extracted nodes and relationships to Neo4j."""

    # 1. Merge User Nodes
    if graph.users:
        user_query = """
        UNWIND $batch AS row
        MERGE (u:User {id: row.id})
        ON CREATE SET u.email = row.email, u.created_at = timestamp()
        ON MATCH SET u.email = row.email
        """
        await neo4j_graph_store.execute_write_batch(user_query, [u.model_dump() for u in graph.users])

    # 2. Merge Project Nodes
    if graph.projects:
        project_query = """
        UNWIND $batch AS row
        MERGE (p:Project {id: row.id})
        ON CREATE SET p.name = row.name, p.description = row.description, p.created_at = timestamp()
        ON MATCH SET p.name = row.name, p.description = row.description
        """
        await neo4j_graph_store.execute_write_batch(project_query, [p.model_dump() for p in graph.projects])

    # 3. Merge Document Nodes
    if graph.documents:
        doc_query = """
        UNWIND $batch AS row
        MERGE (d:Document {id: row.id})
        ON CREATE SET d.name = row.name, d.file_type = row.file_type, d.created_at = timestamp()
        ON MATCH SET d.name = row.name, d.file_type = row.file_type
        """
        await neo4j_graph_store.execute_write_batch(doc_query, [d.model_dump() for d in graph.documents])

    # 4. Merge Requirement Nodes
    if graph.requirements:
        req_query = """
        UNWIND $batch AS row
        MERGE (r:Requirement {id: row.id})
        ON CREATE SET r.title = row.title, r.description = row.description, r.type = row.type, r.priority = row.priority, r.status = row.status, r.created_at = timestamp()
        ON MATCH SET r.title = row.title, r.description = row.description, r.type = row.type, r.priority = row.priority, r.status = row.status
        """
        await neo4j_graph_store.execute_write_batch(req_query, [r.model_dump() for r in graph.requirements])

    # 5. Merge Task Nodes
    if graph.tasks:
        task_query = """
        UNWIND $batch AS row
        MERGE (t:Task {id: row.id})
        ON CREATE SET t.title = row.title, t.description = row.description, t.status = row.status, t.priority = row.priority, t.created_at = timestamp()
        ON MATCH SET t.title = row.title, t.description = row.description, t.status = row.status, t.priority = row.priority
        """
        await neo4j_graph_store.execute_write_batch(task_query, [t.model_dump() for t in graph.tasks])

    # 6. Merge Conflict Nodes
    if graph.conflicts:
        conflict_query = """
        UNWIND $batch AS row
        MERGE (c:Conflict {id: row.id})
        ON CREATE SET c.description = row.description, c.severity = row.severity, c.ai_recommendation = row.ai_recommendation, c.requirement_id = row.requirement_id, c.conflicting_requirement_id = row.conflicting_requirement_id, c.created_at = timestamp()
        ON MATCH SET c.description = row.description, c.severity = row.severity, c.ai_recommendation = row.ai_recommendation
        """
        await neo4j_graph_store.execute_write_batch(conflict_query, [c.model_dump() for c in graph.conflicts])

    # 7. Merge Relationships
    rel_groups: Dict[str, List[Dict[str, Any]]] = {}
    for rel in graph.relationships:
        rel_groups.setdefault(rel.type, []).append(rel.model_dump())

    for rel_type, batch in rel_groups.items():
        if rel_type == "OWNS":
            rel_query = """
            UNWIND $batch AS row
            MATCH (source:User {id: row.source_id})
            MATCH (target:Project {id: row.target_id})
            MERGE (source)-[r:OWNS]->(target)
            """
        elif rel_type == "HAS_DOCUMENT":
            rel_query = """
            UNWIND $batch AS row
            MATCH (source:Project {id: row.source_id})
            MATCH (target:Document {id: row.target_id})
            MERGE (source)-[r:HAS_DOCUMENT]->(target)
            """
        elif rel_type == "TRACKS":
            rel_query = """
            UNWIND $batch AS row
            MATCH (source:Project {id: row.source_id})
            MATCH (target:Requirement {id: row.target_id})
            MERGE (source)-[r:TRACKS]->(target)
            """
        elif rel_type == "CONTAINS":
            rel_query = """
            UNWIND $batch AS row
            MATCH (source:Project {id: row.source_id})
            MATCH (target:Task {id: row.target_id})
            MERGE (source)-[r:CONTAINS]->(target)
            """
        elif rel_type == "IMPLEMENTS":
            rel_query = """
            UNWIND $batch AS row
            MATCH (source:Task {id: row.source_id})
            MATCH (target:Requirement {id: row.target_id})
            MERGE (source)-[r:IMPLEMENTS]->(target)
            """
        elif rel_type == "CONFLICTS_WITH":
            rel_query = """
            UNWIND $batch AS row
            MATCH (source:Requirement {id: row.source_id})
            MATCH (target:Requirement {id: row.target_id})
            MERGE (source)-[r:CONFLICTS_WITH]->(target)
            """
        else:
            logger.warning(f"Skipping unmapped relationship type: {rel_type}")
            continue

        await neo4j_graph_store.execute_write_batch(rel_query, batch)

    # 8. Extra Step: Link Conflict Nodes if present
    if graph.conflicts:
        conflict_link_query = """
        UNWIND $batch AS row
        MATCH (p:Project {id: $project_id})
        MATCH (c:Conflict {id: row.id})
        MATCH (r1:Requirement {id: row.requirement_id})
        MATCH (r2:Requirement {id: row.conflicting_requirement_id})
        MERGE (p)-[:CONTAINS]->(c)
        MERGE (c)-[:AFFECTS]->(r1)
        MERGE (c)-[:AFFECTS]->(r2)
        MERGE (r1)-[cw:CONFLICTS_WITH]->(r2)
        SET cw.severity = row.severity, cw.description = row.description
        """
        await neo4j_graph_store.execute_write_batch(
            conflict_link_query.replace("$project_id", f"'{project_id}'"),
            [c.model_dump() for c in graph.conflicts],
        )

    logger.info(f" Successfully synchronized knowledge graph nodes and relations in Neo4j for project {project_id}")
