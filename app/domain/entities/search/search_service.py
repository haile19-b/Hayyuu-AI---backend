import re
import logging
from typing import List, Optional, Dict, Any

from google.genai import types
from app.core.config import genAI
from app.core.database import prisma
from app.infrastructure.ai.gemini_embedder import gemini_embedder
from app.infrastructure.vector_store.pgvector import pgvector_store
from app.infrastructure.graph_store.neo4j import neo4j_graph_store
from app.domain.entities.search.search_schema import ChatResponse, VectorSource, GraphNodeRef

logger = logging.getLogger("uvicorn.error")

# 1. Define python function signatures for Gemini Tool declarations
def query_project_documents(project_id: str) -> str:
    """Retrieves a list of all documents uploaded to the project, including their id, name, size, and status."""
    return ""

def query_project_requirements(project_id: str) -> str:
    """Retrieves a list of all requirements in the project, including their id, title, type, priority, and status."""
    return ""

def query_project_tasks(project_id: str) -> str:
    """Retrieves a list of all tasks in the project, including their id, title, priority, status, and approval status."""
    return ""

def query_project_conflicts(project_id: str) -> str:
    """Retrieves a list of all active requirement conflicts in the project, including severity, description, and AI recommendation."""
    return ""

def query_project_suggestions(project_id: str) -> str:
    """Retrieves a list of all AI gap analysis suggestions in the project, including type, content, and status."""
    return ""

# List of tools to register with Gemini
POSTGRES_TOOLS = [
    query_project_documents,
    query_project_requirements,
    query_project_tasks,
    query_project_conflicts,
    query_project_suggestions
]

# 2. Async database query runner mapping
async def execute_postgres_tool(name: str, args: dict, default_project_id: str) -> str:
    """Executes the Prisma query corresponding to the function called by Gemini."""
    project_id = args.get("project_id") or args.get("projectId") or default_project_id

    try:
        if name == "query_project_documents":
            docs = await prisma.document.find_many(where={"projectId": project_id})
            if not docs:
                return "No documents found in this project."
            return "\n".join([
                f"- Document '{d.name}' [ID: {d.id}]: Status={d.status}, Type={d.fileType}, Size={d.sizeBytes} bytes"
                for d in docs
            ])

        elif name == "query_project_requirements":
            reqs = await prisma.requirement.find_many(where={"projectId": project_id})
            if not reqs:
                return "No requirements found in this project."
            return "\n".join([
                f"- Requirement '{r.title}' [ID: {r.id}]: Type={r.type}, Priority={r.priority}, Status={r.status}, Conflicted={r.isConflicted}"
                for r in reqs
            ])

        elif name == "query_project_tasks":
            tasks = await prisma.task.find_many(where={"projectId": project_id})
            if not tasks:
                return "No tasks found in this project."
            return "\n".join([
                f"- Task '{t.title}' [ID: {t.id}]: Status={t.status}, Priority={t.priority}, Approval={t.approvalStatus}, Source={t.source}"
                for t in tasks
            ])

        elif name == "query_project_conflicts":
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

        elif name == "query_project_suggestions":
            suggestions = await prisma.aisuggestion.find_many(where={"projectId": project_id})
            if not suggestions:
                return "No suggestions found in this project."
            lines = []
            for s in suggestions:
                content = s.content
                title = content.get("title") if isinstance(content, dict) else "Unnamed Suggestion"
                desc = content.get("description") if isinstance(content, dict) else str(content)
                lines.append(f"- Suggestion '{title}' [ID: {s.id}] (Type: {s.type}): Status={s.status}, Description={desc}")
            return "\n".join(lines)

    except Exception as db_err:
        logger.error(f"Error executing database tool {name}: {db_err}")
        return f"Database query failed: {str(db_err)}"

    return f"Unknown database tool: {name}"


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


class SearchService:
    @staticmethod
    async def retrieve_hybrid_context(
        project_id: str,
        query_text: str,
        document_id: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Performs vector similarity search on PGVector, extracts referenced entity IDs/names,
        and retrieves relevant relationship subgraphs from Neo4j.
        """
        # 1. Generate query embedding & perform vector search
        query_vector = gemini_embedder.embed_text(query_text)
        chunks = await pgvector_store.search_similar(
            query_vector=query_vector,
            project_id=project_id,
            top_k=limit,
            document_id=document_id
        )

        # 2. Extract referenced entities (UUIDs and names)
        entity_ids = set()
        entity_names = set()

        uuid_pattern = re.compile(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}')

        # Match UUIDs in query
        for match in uuid_pattern.findall(query_text):
            entity_ids.add(match)

        # Match UUIDs in chunks content and metadata
        for chunk in chunks:
            content = chunk.get("content", "")
            for match in uuid_pattern.findall(content):
                entity_ids.add(match)

            metadata = chunk.get("metadata") or {}
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

        # 3. Pull Requirement & Task names/IDs from DB to match text mentions
        try:
            requirements = await prisma.requirement.find_many(where={"projectId": project_id})
            tasks = await prisma.task.find_many(where={"projectId": project_id})
        except Exception as db_err:
            logger.error(f"Error fetching project requirements/tasks: {db_err}")
            requirements = []
            tasks = []

        combined_text = (query_text + " " + " ".join([c.get("content", "") for c in chunks])).lower()

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

        # Match any other dynamic nodes in the Neo4j graph for this project by name
        try:
            project_nodes_query = """
            MATCH (p:Project {id: $project_id})-[r]->(n)
            WHERE n.name IS NOT NULL AND NOT n:Document
            RETURN n.id as id, n.name as name, labels(n) as labels
            """
            graph_nodes = await neo4j_graph_store.execute_query(
                project_nodes_query,
                {"project_id": project_id}
            )
            for g_node in graph_nodes:
                g_id = g_node.get("id")
                g_name = g_node.get("name")
                if g_id and g_name:
                    g_name_lower = g_name.lower().strip()
                    if len(g_name_lower) > 3 and g_name_lower in combined_text:
                        entity_ids.add(g_id)
                        entity_names.add(g_name)
        except Exception as graph_err:
            logger.error(f"Error fetching project nodes from Neo4j for name matching: {graph_err}")

        # 4. Traverse Neo4j Graph Database
        rel_records = []
        node_records = []
        if entity_ids or entity_names:
            try:
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
            except Exception as graph_err:
                logger.error(f"Error querying Neo4j graph store: {graph_err}")

        graph_context = format_neo4j_records(rel_records, node_records)

        return {
            "chunks": chunks,
            "rel_records": rel_records,
            "node_records": node_records,
            "graph_context": graph_context
        }

    @staticmethod
    async def generate_rag_answer(
        project_id: str,
        query_text: str,
        document_id: Optional[str] = None,
        limit: int = 5
    ) -> ChatResponse:
        """
        Retrieves combined context, exposes SQL database query tools to Gemini,
        and coordinates tool execution before returning the synthesized response.
        """
        # Retrieve context from PGVector and Neo4j
        context = await SearchService.retrieve_hybrid_context(
            project_id=project_id,
            query_text=query_text,
            document_id=document_id,
            limit=limit
        )

        chunks = context["chunks"]
        rel_records = context["rel_records"]
        node_records = context["node_records"]
        graph_context_str = context["graph_context"]

        # 1. Format document chunks context
        chunks_context = ""
        for idx, chunk in enumerate(chunks):
            chunks_context += f"Source Chunk #{idx + 1} (Chunk ID: {chunk.get('id')}): (Doc ID: {chunk.get('document_id')})\n{chunk.get('content')}\n\n"

        if not chunks_context:
            chunks_context = "No relevant text documents found."

        # 2. Build synthesis prompt
        prompt = f"""
You are Hayyuu AI, a senior systems engineer and product analyst assistant.
Your task is to synthesize a comprehensive, trace-validated answer to the user's question about a project.

Below is the context retrieved from vector search and graph relations:

1. Document Text Chunks (Semantic Vector Search):
---
{chunks_context}
---

2. Graph Relationship Context (Neo4j subgraphs showing dependencies, requirements, tasks, and conflicts):
---
{graph_context_str}
---

Your Active Project ID is: {project_id}
User's Question: {query_text}

Instructions:
1. Synthesize a comprehensive, clear, and factual answer using both the document text chunks and the graph relationships.
2. If you need specific structured database information (like checking document list, listing requirements, checking task statuses, active conflicts, or AI suggestions), you MUST invoke the appropriate database tools provided to you.
3. If there are active conflicts (e.g. RequirementConflict, or CONFLICTS_WITH relationships), explain them clearly along with any recommended solutions.
4. Be specific and trace-validate your answer by referencing entity names and IDs (e.g. [Req ID: req-xxx] or [Task ID: task-xxx]) where appropriate.
5. If the provided context is insufficient to answer the question, state that clearly. Do not make up any facts or relationships.

Synthesized Answer:
"""

        # 3. Perform dynamic tool calling loop
        history = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]

        try:
            logger.info("Initializing multi-turn Gemini call with PostgreSQL tools...")
            response = genAI.models.generate_content(
                model="gemini-3.5-flash",
                contents=history,
                config=types.GenerateContentConfig(
                    tools=POSTGRES_TOOLS,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    temperature=0.0
                )
            )

            # If Gemini requests database tool calls, execute them and run the second turn
            if response.function_calls:
                tool_parts = []
                for call in response.function_calls:
                    logger.info(f"Gemini requested tool call: '{call.name}' with args: {call.args}")
                    result_str = await execute_postgres_tool(call.name, call.args, project_id)
                    tool_parts.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": result_str}
                        )
                    )

                # Append model choice and function response to history
                history.append(response.candidates[0].content)
                history.append(types.Content(role="tool", parts=tool_parts))

                # Re-submit history to get final response
                logger.info("Re-submitting tool results to Gemini for synthesis...")
                response = genAI.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=history,
                    config=types.GenerateContentConfig(
                        tools=POSTGRES_TOOLS,
                        temperature=0.0
                    )
                )

            answer = response.text

        except Exception as llm_err:
            logger.error(f"Error during Gemini synthesis and tool execution: {llm_err}")
            answer = f"Error generating synthesized answer: {str(llm_err)}"

        # 4. Map PGVector results to VectorSource
        sources = []
        for c in chunks:
            sources.append(
                VectorSource(
                    id=c.get("id"),
                    documentId=c.get("document_id"),
                    chunkIndex=c.get("chunk_index"),
                    content=c.get("content"),
                    similarity=c.get("similarity", 0.0),
                    metadata=c.get("metadata")
                )
            )

        # 5. Map Neo4j results to GraphNodeRef
        seen_ids = set()
        nodes = []

        def add_node(node_dict: dict, labels: List[str]):
            n_id = node_dict.get("id")
            if not n_id or n_id in seen_ids:
                return
            seen_ids.add(n_id)
            n_label = labels[0] if labels else "Entity"
            n_name = node_dict.get("name") or node_dict.get("title") or "Unnamed"
            properties = {k: v for k, v in node_dict.items() if k not in ["created_at", "updated_at"]}
            nodes.append(
                GraphNodeRef(
                    id=n_id,
                    label=n_label,
                    name=n_name,
                    properties=properties
                )
            )

        for record in node_records:
            n = record.get("n") or {}
            labels = record.get("n_labels") or []
            add_node(n, labels)

        for record in rel_records:
            n = record.get("n") or {}
            n_labels = record.get("n_labels") or []
            add_node(n, n_labels)
            
            m = record.get("m") or {}
            m_labels = record.get("m_labels") or []
            add_node(m, m_labels)

        return ChatResponse(
            answer=answer,
            sources=sources,
            nodes=nodes
        )
