import logging
import uuid
import re
from typing import Dict, Any
from google.genai import types

from app.core.config import genAI
from app.agents.knowledge_builder.state import KnowledgeBuilderState
from app.agents.knowledge_builder.schema import (
    ExtractedKnowledgeGraph,
    UserNode,
    ProjectNode,
    DocumentNode,
    RequirementNode,
    TaskNode,
    ConflictNode,
    Relationship,
)

logger = logging.getLogger("uvicorn.error")


def normalize_title(title: str) -> str:
    """Normalize title for case-insensitive deterministic ID hashing."""
    val = title.lower().strip()
    val = val.replace("-", " ")
    val = re.sub(r"[^\w\s]", "", val)
    return re.sub(r"\s+", " ", val)


def generate_deterministic_uuid(entity_type: str, project_id: str, unique_str: str) -> str:
    """Generate a stable namespace UUID5 string based on entity type and unique values."""
    normalized = normalize_title(unique_str)
    namespace_str = f"hayyuu:{entity_type}:{project_id}:{normalized}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, namespace_str))


async def extract_knowledge_graph_node(state: KnowledgeBuilderState) -> Dict[str, Any]:
    """
    LLM extraction node that parses raw text and generates structured entities/relations.
    Applies deterministic ID hashing to prevent duplicate nodes.
    """
    prompt = f"""
    You are an expert software engineering analyzer. Your task is to extract requirements, tasks, conflicts, and their relationships from the raw document text.

    Target Schema Details:
    1. Requirement:
       - title: short descriptive title
       - description: detail description of requirement
       - type: FUNCTIONAL or NON_FUNCTIONAL
       - priority: P0, P1, P2
       - status: DRAFT, SUGGESTION, APPROVED, REJECTED
    2. Task:
       - title: short descriptive task title
       - description: description of task
       - status: TODO, IN_PROGRESS, DONE, BLOCKED
       - priority: P0, P1, P2
    3. Conflict (between requirements):
       - description: explain the nature of the conflict
       - severity: HIGH, MEDIUM, LOW
       - ai_recommendation: optional suggestion to resolve the conflict
       - requirement_title: title of the first conflicting requirement
       - conflicting_requirement_title: title of the second conflicting requirement

    Note:
    - Link Tasks to Requirements they implement.
    - Connect requirements that conflict using CONFLICTS_WITH.
    - If User, Project, or Document nodes are not explicitly mentioned in the text, we will synthesize them in the pipeline, but you can output any found or link them.
    - Do not output code block wrappers inside JSON fields.

    Raw Document Context:
    ---
    {state.raw_text}
    ---
    """

    try:
        response = genAI.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedKnowledgeGraph,
                temperature=0.1,
            ),
        )
        raw_graph = ExtractedKnowledgeGraph.model_validate_json(response.text)
    except Exception as e:
        logger.error(f"Failed to generate structured graph from Gemini: {e}")
        return {"errors": state.errors + [f"Gemini generation error: {e}"]}

    project_id = state.project_id
    document_id = state.document_id

    # 1. Synthesize project node
    projects = [ProjectNode(id=project_id, name="Project Context")]
    
    # 2. Synthesize document node if document_id is provided
    documents = []
    if document_id:
        documents.append(DocumentNode(id=document_id, name="Document Source", file_type="pdf"))

    # Map raw LLM generated entity titles/IDs to deterministic stable IDs
    id_mapping: Dict[str, str] = {}

    requirements = []
    for req in raw_graph.requirements:
        stable_id = generate_deterministic_uuid("requirement", project_id, req.title)
        id_mapping[req.id] = stable_id
        # Also map by title fallback
        id_mapping[req.title] = stable_id
        
        requirements.append(
            RequirementNode(
                id=stable_id,
                title=req.title,
                description=req.description,
                type=req.type,
                priority=req.priority,
                status=req.status,
            )
        )

    tasks = []
    for t in raw_graph.tasks:
        stable_id = generate_deterministic_uuid("task", project_id, t.title)
        id_mapping[t.id] = stable_id
        id_mapping[t.title] = stable_id
        
        tasks.append(
            TaskNode(
                id=stable_id,
                title=t.title,
                description=t.description,
                status=t.status,
                priority=t.priority,
            )
        )

    conflicts = []
    for conf in raw_graph.conflicts:
        # Determine IDs of conflicting requirements
        req1_id = id_mapping.get(conf.requirement_id) or generate_deterministic_uuid(
            "requirement", project_id, conf.requirement_id
        )
        req2_id = id_mapping.get(conf.conflicting_requirement_id) or generate_deterministic_uuid(
            "requirement", project_id, conf.conflicting_requirement_id
        )

        stable_conflict_id = generate_deterministic_uuid(
            "conflict", project_id, f"{req1_id}:{req2_id}"
        )
        
        conflicts.append(
            ConflictNode(
                id=stable_conflict_id,
                description=conf.description,
                severity=conf.severity,
                ai_recommendation=conf.ai_recommendation,
                requirement_id=req1_id,
                conflicting_requirement_id=req2_id,
            )
        )

    # Re-map relationships
    relationships = []

    # Synthesize project/doc tracks/contains relations
    for req in requirements:
        relationships.append(Relationship(source_id=project_id, target_id=req.id, type="TRACKS"))
        if document_id:
            relationships.append(Relationship(source_id=document_id, target_id=req.id, type="CONTAINS"))

    for t in tasks:
        relationships.append(Relationship(source_id=project_id, target_id=t.id, type="CONTAINS"))
        if document_id:
            relationships.append(Relationship(source_id=document_id, target_id=t.id, type="CONTAINS"))

    if document_id:
        relationships.append(Relationship(source_id=project_id, target_id=document_id, type="HAS_DOCUMENT"))

    # Map relationships extracted by LLM
    for rel in raw_graph.relationships:
        src_id = id_mapping.get(rel.source_id, rel.source_id)
        tgt_id = id_mapping.get(rel.target_id, rel.target_id)
        
        # If mapping is not a generated UUID, fallback to hash
        if src_id == rel.source_id and rel.source_id not in [project_id, document_id]:
            src_id = generate_deterministic_uuid("requirement", project_id, rel.source_id)
        if tgt_id == rel.target_id and rel.target_id not in [project_id, document_id]:
            tgt_id = generate_deterministic_uuid("requirement", project_id, rel.target_id)

        relationships.append(
            Relationship(
                source_id=src_id,
                target_id=tgt_id,
                type=rel.type,
            )
        )

    # Synthesize direct CONFLICTS_WITH relations from conflicts list
    for conf in conflicts:
        relationships.append(
            Relationship(
                source_id=conf.requirement_id,
                target_id=conf.conflicting_requirement_id,
                type="CONFLICTS_WITH",
            )
        )

    processed_graph = ExtractedKnowledgeGraph(
        users=[],
        projects=projects,
        documents=documents,
        requirements=requirements,
        tasks=tasks,
        conflicts=conflicts,
        relationships=relationships,
    )

    return {"extracted_graph": processed_graph}
