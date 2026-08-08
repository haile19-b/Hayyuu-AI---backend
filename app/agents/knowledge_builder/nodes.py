import logging
import uuid
import re
from typing import Dict, Any, List
from google.genai import types

from app.core.config import genAI
from app.core.database import prisma
from app.core.storage import storage_utility
from app.core.text_extractor import extract_text
from app.libs.chunker import chunk_document_text
from app.infrastructure.ai.gemini_embedder import gemini_embedder
from app.infrastructure.vector_store.pgvector import pgvector_store
from app.infrastructure.graph_store.neo4j import neo4j_graph_store
from app.agents.knowledge_builder.state import KnowledgeBuilderState, ChunkData
from app.agents.knowledge_builder.schema import (
    ExtractedKnowledgeGraph,
    DynamicNode,
    DynamicRelationship,
    PropertyPair,
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
    namespace_str = f"hayyuu:{entity_type.lower()}:{project_id}:{normalized}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, namespace_str))


# 1. Preprocess & Chunk Node
async def preprocess_chunk_node(state: KnowledgeBuilderState) -> Dict[str, Any]:
    """Downloads files if needed, extracts text, and partitions text into token chunks."""
    raw_text = state.raw_text
    doc_id = state.document_id

    if not raw_text.strip():
        if not doc_id:
            return {"errors": state.errors + ["Both raw_text and document_id are missing. Cannot ingest."]}
        
        logger.info(f"[Node: preprocess_chunk] Downloading document '{doc_id}' to extract plain text...")
        try:
            document = await prisma.document.find_unique(where={"id": doc_id})
            if not document:
                return {"errors": state.errors + [f"Document with ID {doc_id} not found."]}

            file_bytes = await storage_utility.download_file(document.filePath)
            raw_text = await extract_text(file_bytes, document.fileType)
        except Exception as e:
            logger.error(f"Error extracting text during preprocessing: {e}")
            return {"errors": state.errors + [f"Text extraction error: {e}"]}

    # Partition text into chunks
    try:
        chunks = chunk_document_text(raw_text)
        chunk_objects = [
            ChunkData(
                chunk_index=c["chunk_index"],
                content=c["content"],
                token_count=c["token_count"],
                start_char=c["start_char"],
                end_char=c["end_char"],
                metadata=c.get("metadata"),
            )
            for c in chunks
        ]
        logger.info(f"[Node: preprocess_chunk] Partitioned text into {len(chunk_objects)} chunks.")
        return {"raw_text": raw_text, "chunks": chunk_objects}
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return {"errors": state.errors + [f"Chunking error: {e}"]}


# 2. Generate Embeddings Node
async def generate_embeddings_node(state: KnowledgeBuilderState) -> Dict[str, Any]:
    """Generates 768-dimensional embeddings for all chunks using gemini-embedding-2."""
    if state.errors:
        return {}

    chunks = state.chunks
    if not chunks:
        return {}

    try:
        logger.info(f"[Node: generate_embeddings] Generating embeddings for {len(chunks)} chunks...")
        contents = [c.content for c in chunks]
        embeddings = gemini_embedder.embed_batch(contents)

        for idx, chunk in enumerate(chunks):
            chunk.embedding = embeddings[idx]

        return {"chunks": chunks}
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return {"errors": state.errors + [f"Embedding generation failed: {e}"]}


# 3. Extract Graph Node
async def extract_graph_node(state: KnowledgeBuilderState) -> Dict[str, Any]:
    """
    Invokes Gemini using structured schemas to extract dynamic entity nodes and relationships.
    Uses stable namespace UUID5 mapping to ensure duplicate-free elements.
    """
    if state.errors:
        return {}

    project_id = state.project_id
    doc_id = state.document_id

    # Segment raw text with chunk index boundaries for source chunk tracking
    chunks_text_list = []
    for c in state.chunks:
        chunks_text_list.append(f"--- Chunk Index: {c.chunk_index} ---\n{c.content}")
    raw_text_with_boundaries = "\n\n".join(chunks_text_list)
    if not raw_text_with_boundaries.strip():
        raw_text_with_boundaries = state.raw_text

    prompt = f"""
    You are an expert systems engineering analyst. Your task is to extract all key entities and relationships from the provided context.

    Guidelines:
    1. Identify all core entities in the document context. Label them appropriately (e.g., 'Requirement', 'Task', 'Conflict', 'Component', 'Actor', etc.).
    2. For each entity you extract, identify which chunk index (0-based integer) it belongs to based on the '--- Chunk Index: X ---' markers in the text, and set the `source_chunk_index` property to that index.
    3. Extract their name, description, and list any other custom attributes (properties) as key-value pairs.
    4. Identify how they relate to each other. Label relationship types descriptively (e.g. 'OWNS', 'TRACKS', 'CONTAINS', 'IMPLEMENTS', 'CONFLICTS_WITH', 'DEPENDS_ON').
    5. Connect the entities using their temporary IDs (the 'id' field you assign, such as 'req_1', 'task_login').
    6. Do not include User, Project, or Document nodes directly in the output list; the pipeline will automatically map and attach them.

    Context to analyze (segmented by chunk boundaries):
    ---
    {raw_text_with_boundaries}
    ---
    """

    try:
        # Check if we should use multimodal file URI or fall back to raw text
        if state.gemini_file_uri and state.gemini_file_mime_type:
            logger.info(f"[Node: extract_graph] Running extraction using Gemini File URI: {state.gemini_file_uri}")
            response = genAI.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    types.Part.from_uri(file_uri=state.gemini_file_uri, mime_type=state.gemini_file_mime_type),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedKnowledgeGraph,
                    temperature=0.1,
                ),
            )
        else:
            logger.info("[Node: extract_graph] Running extraction using raw text input...")
            response = genAI.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedKnowledgeGraph,
                    temperature=0.1,
                ),
            )

        raw_graph = ExtractedKnowledgeGraph.model_validate_json(response.text)
    except Exception as e:
        logger.error(f"Error during graph extraction: {e}")
        return {"errors": state.errors + [f"Graph extraction error: {e}"]}

    # Perform stable ID mapping
    id_mapping: Dict[str, str] = {}
    nodes: List[DynamicNode] = []

    # Map project node
    id_mapping["project"] = project_id
    id_mapping[project_id] = project_id
    
    # Map document node
    if doc_id:
        id_mapping["document"] = doc_id
        id_mapping[doc_id] = doc_id

    # Generate stable IDs for extracted nodes
    for node in raw_graph.nodes:
        # Generate stable UUID
        stable_id = generate_deterministic_uuid(node.label, project_id, node.name)
        id_mapping[node.id] = stable_id
        id_mapping[node.name] = stable_id

        # Update properties list or ensure description is present
        nodes.append(
            DynamicNode(
                id=stable_id,
                label=node.label,
                name=node.name,
                description=node.description,
                source_chunk_index=node.source_chunk_index,
                properties=node.properties,
            )
        )

    # Re-map relationships
    relationships: List[DynamicRelationship] = []

    # 1. Synthesize project level relationships
    for node in nodes:
        # Connect everything to the project
        relationships.append(
            DynamicRelationship(
                source_id=project_id,
                target_id=node.id,
                type="CONTAINS" if node.label.lower() == "task" else "TRACKS",
            )
        )
        if doc_id:
            relationships.append(
                DynamicRelationship(
                    source_id=doc_id,
                    target_id=node.id,
                    type="CONTAINS",
                )
            )

    if doc_id:
        relationships.append(
            DynamicRelationship(
                source_id=project_id,
                target_id=doc_id,
                type="HAS_DOCUMENT",
            )
        )

    # 2. Map relationships extracted by LLM
    for rel in raw_graph.relationships:
        src_id = id_mapping.get(rel.source_id, rel.source_id)
        tgt_id = id_mapping.get(rel.target_id, rel.target_id)

        # If source or target wasn't in our ID mapping, hash it deterministically as a fallback
        if src_id == rel.source_id and rel.source_id not in [project_id, doc_id]:
            src_id = generate_deterministic_uuid("node", project_id, rel.source_id)
        if tgt_id == rel.target_id and rel.target_id not in [project_id, doc_id]:
            tgt_id = generate_deterministic_uuid("node", project_id, rel.target_id)

        relationships.append(
            DynamicRelationship(
                source_id=src_id,
                target_id=tgt_id,
                type=rel.type,
                properties=rel.properties,
            )
        )

    processed_graph = ExtractedKnowledgeGraph(nodes=nodes, relationships=relationships)
    logger.info(f"[Node: extract_graph] Processed {len(nodes)} nodes and {len(relationships)} relationships.")
    return {"extracted_graph": processed_graph}


# 4. Store Vectors Node
async def store_vectors_node(state: KnowledgeBuilderState) -> Dict[str, Any]:
    """Persists document chunks and vector embeddings directly into Neo4j."""
    if state.errors:
        return {}

    doc_id = state.document_id or "synthesized-doc"
    proj_id = state.project_id

    logger.info(f"[Node: store_vectors] Storing vector chunks for document {doc_id} directly in Neo4j...")
    
    import json
    chunk_batch = []
    for c in state.chunks:
        # Generate stable UUID for Chunk node
        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{proj_id}-{doc_id}-chunk-{c.chunk_index}"))
        
        meta = {
            "token_count": c.token_count,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "filename": state.filename,
        }
        if c.metadata:
            meta.update(c.metadata)
            
        chunk_batch.append(
            {
                "id": chunk_uuid,
                "document_id": doc_id,
                "project_id": proj_id,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "embedding": c.embedding,
                "metadata": json.dumps(meta),
            }
        )

    try:
        # Idempotency: delete old chunks for this document in Neo4j
        delete_query = """
        MATCH (c:Chunk {document_id: $doc_id})
        DETACH DELETE c
        """
        await neo4j_graph_store.execute_query(delete_query, {"doc_id": doc_id})

        # Batch insert chunk nodes
        insert_query = """
        UNWIND $batch as row
        MERGE (c:Chunk {id: row.id})
        SET c.document_id = row.document_id,
            c.project_id = row.project_id,
            c.chunk_index = row.chunk_index,
            c.content = row.content,
            c.embedding = row.embedding,
            c.metadata = row.metadata,
            c.created_at = timestamp()
        """
        await neo4j_graph_store.execute_write_batch(insert_query, chunk_batch)

        # Connect chunks to Project
        project_link = """
        MATCH (p:Project {id: $proj_id})
        MATCH (c:Chunk {project_id: $proj_id, document_id: $doc_id})
        MERGE (p)-[:HAS_CHUNK]->(c)
        """
        await neo4j_graph_store.execute_query(project_link, {"proj_id": proj_id, "doc_id": doc_id})

        # Connect chunks to Document
        doc_link = """
        MATCH (d:Document {id: $doc_id})
        MATCH (c:Chunk {document_id: $doc_id})
        MERGE (c)-[:PART_OF]->(d)
        """
        await neo4j_graph_store.execute_query(doc_link, {"doc_id": doc_id})

        # Link chunks sequentially without APOC dependency
        sequence_link = """
        MATCH (c1:Chunk {document_id: $doc_id})
        MATCH (c2:Chunk {document_id: $doc_id})
        WHERE c2.chunk_index = c1.chunk_index + 1
        MERGE (c1)-[:NEXT]->(c2)
        """
        await neo4j_graph_store.execute_query(sequence_link, {"doc_id": doc_id})

        # Link document to the first chunk of each distinct section path
        last_section_path = None
        for c in state.chunks:
            meta = c.metadata or {}
            section_path = meta.get("section_path")
            if section_path and section_path != last_section_path:
                last_section_path = section_path
                chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{proj_id}-{doc_id}-chunk-{c.chunk_index}"))
                section_query = """
                MATCH (d:Document {id: $doc_id})
                MATCH (c:Chunk {id: $chunk_uuid})
                MERGE (d)-[:SECTION {name: $section_name}]->(c)
                """
                await neo4j_graph_store.execute_query(
                    section_query,
                    {
                        "doc_id": doc_id,
                        "chunk_uuid": chunk_uuid,
                        "section_name": section_path
                    }
                )

        return {"total_vectors_stored": len(chunk_batch)}
    except Exception as e:
        logger.error(f"Failed storing chunk vectors in Neo4j: {e}")
        return {"errors": state.errors + [f"Neo4j vector write error: {e}"]}


# 5. Store Graph Node
async def store_graph_node(state: KnowledgeBuilderState) -> Dict[str, Any]:
    """Writes nodes and relationships dynamically into Neo4j."""
    if state.errors:
        return {}

    graph = state.extracted_graph
    project_id = state.project_id
    doc_id = state.document_id

    # 1. Synthesize project node in Neo4j
    project_merge = """
    MERGE (p:Project {id: $id})
    ON CREATE SET p.name = $name, p.created_at = timestamp()
    ON MATCH SET p.name = $name
    """
    try:
        await neo4j_graph_store.execute_query(project_merge, {"id": project_id, "name": f"Project {project_id}"})
        
        # Synthesize document node if document_id is provided
        if doc_id:
            doc_merge = """
            MERGE (d:Document {id: $id})
            ON CREATE SET d.name = $name, d.file_type = $file_type, d.created_at = timestamp()
            ON MATCH SET d.name = $name, d.file_type = $file_type
            """
            await neo4j_graph_store.execute_query(
                doc_merge,
                {"id": doc_id, "name": state.filename, "file_type": state.filename.split(".")[-1]},
            )
    except Exception as e:
        logger.error(f"Failed to merge context nodes in Neo4j: {e}")
        return {"errors": state.errors + [f"Neo4j context merge error: {e}"]}

    # 2. Write dynamic nodes
    node_count = 0
    for node in graph.nodes:
        # Construct Cypher query dynamically using label
        # Safe formatting because label is vetted as alphanumeric/extracted by LLM
        label = re.sub(r"[^\w]", "", node.label) or "Node"
        node_query = f"""
        MERGE (n:Entity:{label} {{id: $id}})
        ON CREATE SET n.name = $name, n.description = $description, n.project_id = $project_id, n.document_id = $document_id, n.created_at = timestamp()
        ON MATCH SET n.name = $name, n.description = $description, n.project_id = $project_id, n.document_id = $document_id
        """
        try:
            await neo4j_graph_store.execute_query(
                node_query,
                {
                    "id": node.id,
                    "name": node.name,
                    "description": node.description,
                    "project_id": project_id,
                    "document_id": doc_id,
                },
            )
            
            # Write dynamic properties
            if node.properties:
                props_dict = {p.key: p.value for p in node.properties}
                props_query = f"""
                MATCH (n:{label} {{id: $id}})
                SET n += $props
                """
                await neo4j_graph_store.execute_query(
                    props_query,
                    {
                        "id": node.id,
                        "props": props_dict,
                    },
                )

            # Link dynamic node to its specific source chunk in Neo4j if available
            if node.source_chunk_index is not None and doc_id:
                chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{project_id}-{doc_id}-chunk-{node.source_chunk_index}"))
                link_query = f"""
                MATCH (c:Chunk {{id: $chunk_uuid}})
                MATCH (n:{label} {{id: $node_id}})
                MERGE (c)-[:HAS_ENTITY]->(n)
                """
                await neo4j_graph_store.execute_query(
                    link_query,
                    {
                        "chunk_uuid": chunk_uuid,
                        "node_id": node.id
                    }
                )

            # Fallback substring matching: link any chunks containing the entity's name
            if doc_id:
                for c in state.chunks:
                    if node.source_chunk_index == c.chunk_index:
                        continue
                    if len(node.name) > 3 and node.name.lower() in c.content.lower():
                        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{project_id}-{doc_id}-chunk-{c.chunk_index}"))
                        fallback_link_query = f"""
                        MATCH (c:Chunk {{id: $chunk_uuid}})
                        MATCH (n:{label} {{id: $node_id}})
                        MERGE (c)-[:MENTIONS]->(n)
                        """
                        await neo4j_graph_store.execute_query(
                            fallback_link_query,
                            {
                                "chunk_uuid": chunk_uuid,
                                "node_id": node.id
                            }
                        )

            node_count += 1
        except Exception as e:
            logger.error(f"Failed writing node '{node.name}' of label '{label}': {e}")
            return {"errors": state.errors + [f"Neo4j node write error: {e}"]}

    # 3. Write dynamic relationships
    rel_count = 0
    for rel in graph.relationships:
        rel_type = re.sub(r"[^\w]", "", rel.type) or "RELATED_TO"
        
        # Query dynamically matching any source/target nodes by their unique ID
        rel_query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        """
        try:
            await neo4j_graph_store.execute_query(
                rel_query,
                {"source_id": rel.source_id, "target_id": rel.target_id},
            )

            # Write relationship properties
            if rel.properties:
                props_dict = {p.key: p.value for p in rel.properties}
                rel_props_query = f"""
                MATCH (source {{id: $source_id}})-[r:{rel_type}]->(target {{id: $target_id}})
                SET r += $props
                """
                await neo4j_graph_store.execute_query(
                    rel_props_query,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "props": props_dict,
                    },
                )
            rel_count += 1
        except Exception as e:
            logger.error(f"Failed writing relationship type '{rel_type}' from '{rel.source_id}' to '{rel.target_id}': {e}")
            # Non-blocking error, log and continue
            logger.warning(f"Could not connect source {rel.source_id} to target {rel.target_id} in Neo4j.")

    logger.info(f"[Node: store_graph] Wrote {node_count} nodes and {rel_count} relations to Neo4j.")
    return {"total_nodes_stored": node_count + rel_count}


# 6. Save Results Node
async def save_results_node(state: KnowledgeBuilderState) -> Dict[str, Any]:
    """Compiles statistics and saves results status."""
    if state.errors:
        logger.error(f"Knowledge builder pipeline finished with errors: {state.errors}")
        return {"status": "failed"}

    logger.info(f"Knowledge builder pipeline completed. Vectors stored: {state.total_vectors_stored}, Graph elements stored: {state.total_nodes_stored}")
    return {"status": "completed"}
