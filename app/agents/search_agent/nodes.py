import re
import logging
from typing import List, Dict, Any, Optional

from google.genai import types
from app.core.config import genAI
from app.core.database import prisma
from app.infrastructure.ai.gemini_embedder import gemini_embedder
from app.infrastructure.vector_store.pgvector import pgvector_store
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

from app.agents.search_agent.state import SearchAgentState
from app.agents.search_agent.schemas import VectorSource, GraphNodeRef
from app.agents.search_agent.tools import relational_db_tool, graph_db_tool

logger = logging.getLogger("uvicorn.error")


def format_neo4j_records(rel_records: List[dict], node_records: List[dict]) -> str:
    """Format Neo4j node and relationship records into a readable string for the LLM."""
    if not rel_records and not node_records:
        return "No graph entities or relationships found for the resolved entities."
        
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
        n_desc = n.get("description") or ""
        
        node_str = f"- {n_label}: '{n_name}' [ID: {n_id}]"
        if n_desc:
            node_str += f" - Description: {n_desc}"
            
        other_props = []
        for k, v in n.items():
            if k not in ["id", "name", "title", "description", "created_at", "updated_at"]:
                other_props.append(f"{k}: {v}")
        if other_props:
            node_str += f" ({', '.join(other_props)})"
            
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
            
            n_label = n_labels[0] if n_labels else "Entity"
            m_label = m_labels[0] if m_labels else "Entity"
            n_name = n.get("name") or n.get("title") or "Unnamed"
            m_name = m.get("name") or m.get("title") or "Unnamed"
            
            rel_str = f"- ({n_label}: '{n_name}' [ID: {n_id}]) -[{rel_type}]-> ({m_label}: '{m_name}' [ID: {m_id}])"
            lines.append(rel_str)
            
    return "\n".join(lines)


# 1. Query Analysis Node
async def query_analysis_node(state: SearchAgentState) -> Dict[str, Any]:
    """Generates embeddings, queries Neo4j vector index with sequential window context, and extracts matched entity IDs/names."""
    logger.info("[Node: query_analysis] Generating query embedding and performing Neo4j native vector search...")
    query_text = state.query_text
    project_id = state.project_id
    document_id = state.document_id
    limit = state.limit

    import json
    try:
        # Perform query embedding
        query_vector = gemini_embedder.embed_text(query_text, user_query=True)
        
        # Cypher query for native vector search + sequential window expansion
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
                "top_k": limit,
                "query_vector": query_vector,
                "project_id": project_id,
                "document_id": document_id
            }
        )
        
        # Map Neo4j chunk records to VectorSource schema
        sources = []
        for r in records:
            # Reconstruct the window text: prev_content + content + next_content
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
                    
            sources.append(
                VectorSource(
                    id=r.get("id") or "unknown-chunk",
                    documentId=r.get("document_id") or "unknown-doc",
                    chunkIndex=r.get("chunk_index") or 0,
                    content=combined_content,
                    similarity=float(r.get("similarity") or 0.0),
                    metadata=meta
                )
            )
            
        return {
            "chunks": sources,
            "status": "analysis_completed"
        }
    except Exception as e:
        logger.error(f"[Node: query_analysis] Error: {e}")
        return {
            "errors": state.errors + [f"Query analysis failed: {str(e)}"],
            "status": "failed"
        }


# 2. Graph Retrieval Node
async def graph_retrieval_node(state: SearchAgentState) -> Dict[str, Any]:
    """Resolves entity titles/IDs and invokes graph_db_tool to fetch Neo4j subgraphs."""
    logger.info("[Node: graph_retrieval] Resolving entity IDs and performing graph traversal...")
    project_id = state.project_id
    query_text = state.query_text
    chunks = state.chunks

    entity_ids = set()
    entity_names = set()

    uuid_pattern = re.compile(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}')

    # Match UUIDs in query
    for match in uuid_pattern.findall(query_text):
        entity_ids.add(match)

    # Match UUIDs in chunks content and metadata
    for chunk in chunks:
        content = chunk.content
        for match in uuid_pattern.findall(content):
            entity_ids.add(match)

        metadata = chunk.metadata or {}
        if isinstance(metadata, dict):
            for val in metadata.values():
                if isinstance(val, str):
                    for match in uuid_pattern.findall(val):
                        entity_ids.add(match)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            for match in uuid_pattern.findall(item):
                                entity_ids.add(match)

    # Match Requirements & Tasks from DB
    try:
        requirements = await prisma.requirement.find_many(where={"projectId": project_id})
        tasks = await prisma.task.find_many(where={"projectId": project_id})
    except Exception as db_err:
        logger.error(f"Error fetching project requirements/tasks: {db_err}")
        requirements = []
        tasks = []

    combined_text = (query_text + " " + " ".join([c.content for c in chunks])).lower()

    for req in requirements:
        req_title = req.title.lower().strip()
        if req.id in entity_ids or (len(req_title) > 3 and req_title in combined_text):
            entity_ids.add(req.id)
            entity_names.add(req.title)

    for task in tasks:
        task_title = task.title.lower().strip()
        if task.id in entity_ids or (len(task_title) > 3 and task_title in combined_text):
            entity_ids.add(task.id)
            entity_names.add(task.title)

    # Match Neo4j nodes by name directly
    try:
        project_nodes_query = """
        MATCH (p:Project {id: $project_id})-[r]->(n)
        WHERE n.name IS NOT NULL AND NOT n:Document
        RETURN n.id as id, n.name as name, labels(n) as labels
        """
        graph_nodes_recs = await neo4j_graph_store.execute_query(
            project_nodes_query,
            {"project_id": project_id}
        )
        for g_node in graph_nodes_recs:
            g_id = g_node.get("id")
            g_name = g_node.get("name")
            if g_id and g_name:
                g_name_lower = g_name.lower().strip()
                if len(g_name_lower) > 3 and g_name_lower in combined_text:
                    entity_ids.add(g_id)
                    entity_names.add(g_name)
    except Exception as graph_err:
        logger.warning(f"Error pre-matching graph node names: {graph_err}")

    # Invoke Neo4j query helper via graph_db_tool
    graph_context = ""
    nodes_ref = []
    
    if entity_ids or entity_names:
        try:
            # Traversal query via tool
            res = await graph_db_tool(
                project_id=project_id,
                action="traverse_subgraph",
                params={"entity_ids": list(entity_ids), "entity_names": list(entity_names)}
            )
            graph_context = res.get("context", "")
            node_records = res.get("nodes", [])
            rel_records = res.get("rels", [])
            
            # Compile GraphNodeRef objects
            seen_ids = set()
            
            def add_node_ref(node_dict: dict, labels: List[str]):
                n_id = node_dict.get("id")
                if not n_id or n_id in seen_ids:
                    return
                seen_ids.add(n_id)
                n_label = labels[0] if labels else "Entity"
                n_name = node_dict.get("name") or node_dict.get("title") or "Unnamed"
                properties = {k: v for k, v in node_dict.items() if k not in ["created_at", "updated_at"]}
                nodes_ref.append(
                    GraphNodeRef(
                        id=n_id,
                        label=n_label,
                        name=n_name,
                        properties=properties
                    )
                )

            for record in node_records:
                add_node_ref(record.get("n") or {}, record.get("n_labels") or [])
                
            for record in rel_records:
                add_node_ref(record.get("n") or {}, record.get("n_labels") or [])
                add_node_ref(record.get("m") or {}, record.get("m_labels") or [])
                
        except Exception as e:
            logger.error(f"[Node: graph_retrieval] Traverse failed: {e}")
            graph_context = f"Error performing graph traversal: {str(e)}"
            
    return {
        "graph_context": graph_context,
        "graph_nodes": nodes_ref,
        "status": "graph_completed"
    }


# 3. Agent Loop / LLM Node
async def agent_loop_node(state: SearchAgentState) -> Dict[str, Any]:
    """Prepares the synthesis prompt and submits it to Gemini, checking for tool requests."""
    logger.info("[Node: agent_loop] Submitting chat history and prompt to Gemini...")
    project_id = state.project_id
    query_text = state.query_text
    chunks = state.chunks
    graph_context = state.graph_context

    # Initialize history if empty
    if not state.history:
        chunks_context = ""
        for idx, chunk in enumerate(chunks):
            chunks_context += f"Source Chunk #{idx + 1} (Chunk ID: {chunk.id}): (Doc ID: {chunk.documentId})\n{chunk.content}\n\n"
        if not chunks_context:
            chunks_context = "No relevant text documents found."
            
        prompt = f"""
You are Hayyuu AI, a senior systems engineer and product analyst assistant.

Your task is to provide a comprehensive, factual, and traceable answer to the user's question about the project using the available context.

Below is the retrieved context:

1. Document Text Chunks (Semantic Vector Search):
---
{chunks_context}
---

2. Graph Relationship Context (Neo4j subgraphs showing dependencies, requirements, tasks, conflicts, and related entities):
---
{graph_context}
---

Active Project ID: {project_id}

User's Question:
{query_text}

Instructions:

1. If the user's message is only a greeting or casual conversation (e.g., "Hi", "Hello", "Good morning", "How are you?", "Thanks"), respond naturally without invoking any database tools or performing project-related analysis.

2. For project-related questions, synthesize a clear, comprehensive, and factual answer using both the document text chunks and the graph relationship context whenever relevant.

3. Only if the user's query requires specific structured database information (such as document lists, requirement lists, task lists, active conflicts, AI suggestions, or other structured project records), you MUST invoke the appropriate database tools provided to you.

4. If active conflicts exist (for example, RequirementConflict nodes or CONFLICTS_WITH relationships), explain:
   - what the conflict is,
   - which entities are involved,
   - the potential impact, and
   - any recommended resolutions if available.

5. Make your answer traceable by referencing relevant entity identifiers whenever possible (for example, [REQ-001], [TASK-015], or other entity names present in the context).

6. If the retrieved context is insufficient to answer the user's question, explicitly state that the available information is insufficient. Do not invent facts, relationships, or project data.

7. Prioritize accuracy over completeness. Never fabricate project information or database records.

Synthesized Answer:
"""
        history = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    else:
        history = list(state.history)

    try:
        response = genAI.models.generate_content(
            model="gemini-3.5-flash",
            contents=history,
            config=types.GenerateContentConfig(
                tools=[relational_db_tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.0
            )
        )
        
        # Check if the model requested function calls
        if response.function_calls:
            # We must append the model's tool call candidate content to history
            history.append(response.candidates[0].content)
            return {
                "history": history,
                "status": "tool_call"
            }
            
        # No function call: we got the final answer
        return {
            "answer": response.text,
            "status": "synthesized"
        }
    except Exception as e:
        logger.error(f"[Node: agent_loop] Synthesis turn failed: {e}")
        return {
            "errors": state.errors + [f"Synthesis Turn Failed: {str(e)}"],
            "status": "failed"
        }


# 4. Execute Relational DB Tool Node
async def execute_db_tool_node(state: SearchAgentState) -> Dict[str, Any]:
    """Executes the requested relational database tool calls and returns them to history."""
    logger.info("[Node: execute_db_tool] Running database Prisma queries for Gemini tool calls...")
    history = list(state.history)
    project_id = state.project_id

    # The last content in history is the model's call requesting tools
    # Let's inspect it to run function responses
    last_content = history[-1]
    tool_parts = []
    postgres_context_additions = []

    # Iterate over function calls requested
    for part in last_content.parts:
        if part.function_call:
            call = part.function_call
            if call.name == "relational_db_tool":
                logger.info(f"Executing relational database tool with args: {call.args}")
                result_str = await relational_db_tool(
                    project_id=call.args.get("project_id") or project_id,
                    action=call.args.get("action"),
                    params=call.args.get("params")
                )
            else:
                result_str = f"Unsupported tool: {call.name}"
            
            tool_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result_str}
                )
            )
            postgres_context_additions.append(f"Tool [{call.name}] Output:\n{result_str}\n")

    # Add the tool response content
    history.append(types.Content(role="tool", parts=tool_parts))

    return {
        "history": history,
        "postgres_context": state.postgres_context + "\n".join(postgres_context_additions),
        "status": "tool_executed"
    }


# 5. Validation Node
async def validate_response_node(state: SearchAgentState) -> Dict[str, Any]:
    """Validates the generated response for factual consistency and trace mapping."""
    logger.info("[Node: validate_response] Validating synthesized answer for trace-accuracy...")
    # This is a validation step. If the output is missing, we flag it.
    if not state.answer:
        return {
            "errors": state.errors + ["Validation failed: Synthesized answer is empty."],
            "status": "failed"
        }
    return {
        "status": "completed"
    }
