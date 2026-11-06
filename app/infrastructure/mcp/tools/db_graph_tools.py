import logging
from typing import List, Optional, Any, Dict
from mcp.server import MCPServer
from prisma.enums import (
    RequirementType,
    RequirementPriority,
    RequirementStatus,
    TaskStatus,
    TaskPriority,
    TaskApprovalStatus,
    TaskSource,
    SuggestionStatus,
    ConflictSeverity,
    ConflictStatus,
)

from app.core.database import prisma
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

logger = logging.getLogger("uvicorn.error")


def register_db_graph_tools(mcp: MCPServer) -> None:
    """Registers core database and Neo4j graph tools on the given MCPServer instance."""

    @mcp.tool()
    async def list_project_documents(
        project_id: str,
        status: Optional[str] = None,
        file_type: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """
        Retrieves metadata of uploaded documents in the project with optional filtering.

        Args:
            project_id: The UUID of the active project.
            status: Optional filter by status: 'PENDING', 'INDEXED', or 'FAILED'.
            file_type: Optional filter by file extension/type (e.g. 'pdf', 'docx', 'txt').
            search_query: Optional keyword to match against document names.
            limit: Maximum documents to return (default 50).
        """
        logger.info(f"[mcp: list_project_documents] Executing for project '{project_id}'")
        try:
            where: dict = {"projectId": project_id}
            if status:
                s_norm = status.upper().strip()
                if s_norm in ["PENDING", "INDEXED", "FAILED"]:
                    where["status"] = s_norm
            if file_type:
                where["fileType"] = {"equals": file_type.lower().strip()}
            if search_query:
                where["name"] = {"contains": search_query.strip(), "mode": "insensitive"}

            docs = await prisma.document.find_many(where=where, take=limit)
            if not docs:
                return "No documents found matching the criteria."
            return "\n".join([
                f"- Document '{d.name}' [ID: {d.id}]: Status={d.status}, Type={d.fileType}, Size={d.sizeBytes} bytes"
                for d in docs
            ])
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return f"Error listing documents: {str(e)}"

    @mcp.tool()
    async def list_project_requirements(
        project_id: str,
        type: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        is_conflicted: Optional[bool] = None,
        search_query: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """
        Retrieves requirements in the project with rich filtering for focused retrieval.

        Args:
            project_id: The UUID of the active project.
            type: Filter by requirement type: 'FUNCTIONAL' or 'NON_FUNCTIONAL' (case-insensitive).
            status: Filter by requirement status: 'DRAFT', 'SUGGESTION', 'APPROVED', 'REJECTED'.
            priority: Filter by priority level: 'P0' (Critical), 'P1' (High), 'P2' (Medium).
            is_conflicted: Filter by conflict state: True for conflicted requirements, False for non-conflicted.
            search_query: Optional keyword to search within title and description.
            limit: Maximum requirements to return (default 50).
        """
        logger.info(f"[mcp: list_project_requirements] Executing for project '{project_id}'")
        try:
            where: dict = {"projectId": project_id}
            if type:
                t_norm = type.upper().replace("-", "_").strip()
                if t_norm in ["FUNCTIONAL", "NON_FUNCTIONAL"]:
                    where["type"] = t_norm
            if status:
                s_norm = status.upper().replace(" ", "_").strip()
                if s_norm in ["DRAFT", "SUGGESTION", "APPROVED", "REJECTED"]:
                    where["status"] = s_norm
            if priority:
                p_norm = priority.upper().strip()
                if p_norm in ["P0", "P1", "P2"]:
                    where["priority"] = p_norm
            if is_conflicted is not None:
                where["isConflicted"] = is_conflicted
            if search_query:
                q = search_query.strip()
                where["OR"] = [
                    {"title": {"contains": q, "mode": "insensitive"}},
                    {"description": {"contains": q, "mode": "insensitive"}},
                ]

            reqs = await prisma.requirement.find_many(where=where, take=limit)
            if not reqs:
                return "No requirements found matching the criteria."
            return "\n".join([
                f"- Requirement '{r.title}' [ID: {r.id}]: Type={r.type}, Priority={r.priority}, Status={r.status}, Conflicted={r.isConflicted}\n"
                f"  Summary: {r.description[:120]}..."
                for r in reqs
            ])
        except Exception as e:
            logger.error(f"Error listing requirements: {e}")
            return f"Error listing requirements: {str(e)}"

    @mcp.tool()
    async def list_project_tasks(
        project_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        approval_status: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """
        Retrieves developer tasks in the project with status, priority, and text search filters.

        Args:
            project_id: The UUID of the active project.
            status: Filter by status: 'TODO', 'IN_PROGRESS', 'DONE', 'BLOCKED'.
            priority: Filter by priority: 'P0' (Critical), 'P1' (High), 'P2' (Medium).
            approval_status: Filter by approval status: 'PENDING', 'APPROVED', 'REJECTED'.
            search_query: Optional keyword to match against task title and description.
            limit: Maximum tasks to return (default 50).
        """
        logger.info(f"[mcp: list_project_tasks] Executing for project '{project_id}'")
        try:
            where: dict = {"projectId": project_id}
            if status:
                s_norm = status.upper().replace(" ", "_").strip()
                if s_norm in ["TODO", "IN_PROGRESS", "DONE", "BLOCKED"]:
                    where["status"] = s_norm
            if priority:
                p_norm = priority.upper().strip()
                if p_norm in ["P0", "P1", "P2"]:
                    where["priority"] = p_norm
            if approval_status:
                a_norm = approval_status.upper().strip()
                if a_norm in ["PENDING", "APPROVED", "REJECTED"]:
                    where["approvalStatus"] = a_norm
            if search_query:
                q = search_query.strip()
                where["OR"] = [
                    {"title": {"contains": q, "mode": "insensitive"}},
                    {"description": {"contains": q, "mode": "insensitive"}},
                ]

            tasks = await prisma.task.find_many(where=where, take=limit)
            if not tasks:
                return "No tasks found matching the criteria."
            return "\n".join([
                f"- Task '{t.title}' [ID: {t.id}]: Status={t.status}, Priority={t.priority}, Approval={t.approvalStatus}, Source={t.source}"
                for t in tasks
            ])
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return f"Error listing tasks: {str(e)}"

    @mcp.tool()
    async def list_project_conflicts(
        project_id: str,
        status: Optional[str] = "ACTIVE",
        severity: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """
        Retrieves requirement conflicts in the project with severity and status filtering.

        Args:
            project_id: The UUID of the active project.
            status: Filter by status: 'ACTIVE', 'RESOLVED', or 'ALL' (default 'ACTIVE').
            severity: Optional filter by severity: 'HIGH', 'MEDIUM', 'LOW'.
            limit: Maximum conflicts to return (default 50).
        """
        logger.info(f"[mcp: list_project_conflicts] Executing for project '{project_id}'")
        try:
            where: dict = {"projectId": project_id}
            if status and status.upper().strip() != "ALL":
                s_norm = status.upper().strip()
                if s_norm in ["ACTIVE", "RESOLVED"]:
                    where["status"] = s_norm
            if severity:
                sev_norm = severity.upper().strip()
                if sev_norm in ["HIGH", "MEDIUM", "LOW"]:
                    where["severity"] = sev_norm

            conflicts = await prisma.requirementconflict.find_many(
                where=where,
                take=limit,
                include={"requirement": True, "conflictingRequirement": True}
            )
            if not conflicts:
                return "No conflicts found matching the criteria."
            return "\n".join([
                f"- Conflict [ID: {c.id}] (Status: {c.status}, Severity: {c.severity}): Description={c.description}, Recommendation={c.aiRecommendation or 'None'}, "
                f"Requirements involved: Requirement A ID={c.requirementId} ({c.requirement.title if c.requirement else 'Unknown'}), "
                f"Requirement B ID={c.conflictingRequirementId} ({c.conflictingRequirement.title if c.conflictingRequirement else 'Unknown'})"
                for c in conflicts
            ])
        except Exception as e:
            logger.error(f"Error listing conflicts: {e}")
            return f"Error listing conflicts: {str(e)}"

    @mcp.tool()
    async def list_project_suggestions(
        project_id: str,
        status: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """
        Retrieves AI suggested requirements and gaps for the project with status and type filtering.

        Args:
            project_id: The UUID of the active project.
            status: Optional filter by status: 'PENDING', 'APPROVED', 'REJECTED'.
            type: Optional filter by suggestion type.
            limit: Maximum suggestions to return (default 50).
        """
        logger.info(f"[mcp: list_project_suggestions] Executing for project '{project_id}'")
        try:
            where: dict = {"projectId": project_id}
            if status:
                s_norm = status.upper().strip()
                if s_norm in ["PENDING", "APPROVED", "REJECTED"]:
                    where["status"] = s_norm
            if type:
                where["type"] = type.strip()

            suggestions = await prisma.aisuggestion.find_many(where=where, take=limit)
            if not suggestions:
                return "No AI suggestions found matching the criteria."
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

    @mcp.tool()
    async def get_chat_history(
        thread_id: str,
        limit: int = 30,
        before_message_id: Optional[str] = None,
        after_message_id: Optional[str] = None
    ) -> str:
        """
        Retrieves sequential conversation turns for a thread ordered chronologically.
        Provides cursor-based window pagination via before_message_id or after_message_id.

        Args:
            thread_id: The UUID of the conversation thread.
            limit: Maximum messages to return (default 30, capped at 100).
            before_message_id: Optional UUID cursor to return messages created strictly before this message.
            after_message_id: Optional UUID cursor to return messages created strictly after this message.
        """
        logger.info(f"[mcp: get_chat_history] Executing for thread '{thread_id}' (limit {limit})")
        try:
            limit = min(max(1, limit), 100)
            where: dict = {"threadId": thread_id}

            if before_message_id:
                ref = await prisma.chatmessage.find_unique(where={"id": before_message_id})
                if ref:
                    where["createdAt"] = {"lt": ref.createdAt}
            elif after_message_id:
                ref = await prisma.chatmessage.find_unique(where={"id": after_message_id})
                if ref:
                    where["createdAt"] = {"gt": ref.createdAt}

            messages = await prisma.chatmessage.find_many(
                where=where,
                order={"createdAt": "asc"},
                take=limit
            )

            if not messages:
                return f"No chat history found for thread '{thread_id}' with the given constraints."

            formatted = []
            for msg in messages:
                role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role).upper()
                action_str = f" [action: {msg.cognitiveAction}]" if msg.cognitiveAction else ""
                style_str = f" [style: {msg.styleModifier}]" if msg.styleModifier else ""
                formatted.append(f"[{msg.createdAt.strftime('%Y-%m-%d %H:%M:%S')}] ({msg.id}) {role_str}{action_str}{style_str}:\n{msg.content}\n")

            return "\n---\n".join(formatted)
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            return f"Error fetching chat history: {str(e)}"

    @mcp.tool()
    async def search_chat_history(
        query: str,
        project_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 5,
        semantic: bool = True
    ) -> str:
        """
        Performs semantic vector search or keyword search across messages in a chat thread or across an entire project.

        Args:
            query: Search query string.
            project_id: Optional UUID of the project to search across all threads.
            thread_id: Optional UUID of a specific conversation thread.
            limit: Maximum matches to return (default 5, max 20).
            semantic: If True, uses vector cosine similarity via 768-dim embeddings; falls back to text search if no embeddings exist.
        """
        logger.info(f"[mcp: search_chat_history] Executing search (query='{query}', project_id={project_id}, thread_id={thread_id}, semantic={semantic})")
        if not project_id and not thread_id:
            return "Error: At least one of 'project_id' or 'thread_id' must be provided to search chat history."

        try:
            limit = min(max(1, limit), 20)
            results = []

            if semantic:
                try:
                    from app.infrastructure.ai.gemini_embedder import GeminiEmbedder
                    embedder = GeminiEmbedder()
                    query_vec = embedder.embed_text(query, user_query=True)
                    vec_literal = "[" + ",".join(str(x) for x in query_vec) + "]"

                    if thread_id:
                        raw_res = await prisma.query_raw(
                            f"""
                            SELECT id, role, content, "cognitiveAction", "createdAt", "threadId",
                                   (1 - (embedding <=> '{vec_literal}'::vector)) AS similarity
                            FROM chat_messages
                            WHERE "threadId" = $1 AND embedding IS NOT NULL
                            ORDER BY embedding <=> '{vec_literal}'::vector ASC
                            LIMIT $2;
                            """,
                            thread_id,
                            limit
                        )
                    else:
                        raw_res = await prisma.query_raw(
                            f"""
                            SELECT m.id, m.role, m.content, m."cognitiveAction", m."createdAt", m."threadId",
                                   (1 - (m.embedding <=> '{vec_literal}'::vector)) AS similarity
                            FROM chat_messages m
                            JOIN conversation_threads t ON m."threadId" = t.id
                            WHERE t."projectId" = $1 AND m.embedding IS NOT NULL
                            ORDER BY m.embedding <=> '{vec_literal}'::vector ASC
                            LIMIT $2;
                            """,
                            project_id,
                            limit
                        )

                    if raw_res:
                        for r in raw_res:
                            sim = float(r.get("similarity") or 0.0)
                            results.append(f"Match (Score: {sim:.3f}) [{r.get('role')}] ({r.get('id')} in thread {r.get('threadId')}): {r.get('content')}")
                except Exception as embed_err:
                    logger.warning(f"Semantic search failed, falling back to keyword search: {embed_err}")

            if not results:
                # Text fallback (ILIKE search)
                where: dict = {"content": {"contains": query.strip(), "mode": "insensitive"}}
                if thread_id:
                    where["threadId"] = thread_id
                elif project_id:
                    where["thread"] = {"projectId": project_id}

                matches = await prisma.chatmessage.find_many(
                    where=where,
                    order={"createdAt": "desc"},
                    take=limit
                )
                for m in matches:
                    role_str = m.role.value if hasattr(m.role, "value") else str(m.role).upper()
                    results.append(f"Keyword Match [{role_str}] ({m.id} in thread {m.threadId}): {m.content[:300]}")

            if not results:
                target_scope = f"thread '{thread_id}'" if thread_id else f"project '{project_id}'"
                return f"No messages found matching query '{query}' in {target_scope}."

            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Error searching chat history: {e}")
            return f"Error searching chat history: {str(e)}"

    @mcp.tool()
    async def get_requirement_details(requirement_id: str) -> str:
        """
        Retrieves full specification, conflicts, and tasks for a specific requirement UUID.

        Args:
            requirement_id: UUID of the requirement.
        """
        try:
            req = await prisma.requirement.find_unique(
                where={"id": requirement_id},
                include={
                    "conflicts": True,
                    "conflictedBy": True,
                    "tasks": True,
                    "project": True,
                }
            )
            if not req:
                return f"Error: Requirement '{requirement_id}' not found."

            conflicts_info = []
            for c in (req.conflicts + req.conflictedBy):
                conflicts_info.append(f"- Conflict [ID: {c.id}]: Severity={c.severity}, Status={c.status}, Details: {c.description}")
            conflicts_str = "\n".join(conflicts_info) if conflicts_info else "None"

            tasks_info = [
                f"- Task '{t.title}' [ID: {t.id}]: Status={t.status}, Priority={t.priority}"
                for t in req.tasks
            ]
            tasks_str = "\n".join(tasks_info) if tasks_info else "None"

            return (
                f"Requirement: '{req.title}' [ID: {req.id}]\n"
                f"- Project: {req.project.name if req.project else req.projectId}\n"
                f"- Type: {req.type}\n"
                f"- Priority: {req.priority}\n"
                f"- Status: {req.status}\n"
                f"- Conflicted: {req.isConflicted}\n"
                f"- Description: {req.description}\n"
                f"- Linked Tasks:\n{tasks_str}\n"
                f"- Associated Conflicts:\n{conflicts_str}"
            )
        except Exception as e:
            logger.error(f"Error fetching requirement details: {e}")
            return f"Error fetching requirement details: {str(e)}"

    @mcp.tool()
    async def update_task_details(
        task_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        approval_status: Optional[str] = None
    ) -> str:
        """
        Updates attributes of an existing project task.

        Args:
            task_id: UUID of the task to update.
            status: Optional new status ('TODO', 'IN_PROGRESS', 'DONE', 'BLOCKED').
            priority: Optional new priority ('P0', 'P1', 'P2').
            approval_status: Optional new approval status ('PENDING', 'APPROVED', 'REJECTED').
        """
        try:
            from prisma.enums import TaskStatus, TaskPriority, TaskApprovalStatus
            data: dict = {}
            if status:
                s_norm = status.upper().replace(" ", "_").strip()
                if s_norm in ["TODO", "IN_PROGRESS", "DONE", "BLOCKED"]:
                    data["status"] = TaskStatus[s_norm]
            if priority:
                p_norm = priority.upper().strip()
                if p_norm in ["P0", "P1", "P2"]:
                    data["priority"] = TaskPriority[p_norm]
            if approval_status:
                a_norm = approval_status.upper().strip()
                if a_norm in ["PENDING", "APPROVED", "REJECTED"]:
                    data["approvalStatus"] = TaskApprovalStatus[a_norm]

            if not data:
                return "No valid update fields provided."

            task = await prisma.task.update(
                where={"id": task_id},
                data=data
            )
            if not task:
                return f"Error: Task '{task_id}' not found."

            return f"Task '{task.title}' [ID: {task.id}] successfully updated. Status={task.status}, Priority={task.priority}, Approval={task.approvalStatus}"
        except Exception as e:
            logger.error(f"Error updating task: {e}")
            return f"Error updating task: {str(e)}"

    @mcp.tool()
    async def resolve_requirement_conflict(
        conflict_id: str,
        resolution_notes: str,
        chosen_requirement_id: Optional[str] = None
    ) -> str:
        """
        Resolves an active requirement conflict and updates status in the database.

        Args:
            conflict_id: UUID of the requirement conflict.
            resolution_notes: Explanatory notes or compromise agreed upon for resolution.
            chosen_requirement_id: Optional UUID of the requirement that takes precedence.
        """
        try:
            conflict = await prisma.requirementconflict.find_unique(where={"id": conflict_id})
            if not conflict:
                return f"Error: Conflict '{conflict_id}' not found."

            await prisma.requirementconflict.update(
                where={"id": conflict_id},
                data={
                    "status": "RESOLVED",
                    "aiRecommendation": resolution_notes
                }
            )

            # Clear isConflicted flag if no other active conflicts exist for these requirements
            for req_id in [conflict.requirementId, conflict.conflictingRequirementId]:
                remaining = await prisma.requirementconflict.count(
                    where={
                        "status": "ACTIVE",
                        "OR": [
                            {"requirementId": req_id},
                            {"conflictingRequirementId": req_id}
                        ]
                    }
                )
                if remaining == 0:
                    await prisma.requirement.update(
                        where={"id": req_id},
                        data={"isConflicted": False}
                    )

            return f"Conflict [ID: {conflict_id}] resolved successfully. Resolution: {resolution_notes}"
        except Exception as e:
            logger.error(f"Error resolving conflict: {e}")
            return f"Error resolving conflict: {str(e)}"

    @mcp.tool()
    async def list_project_repositories(project_id: str) -> str:
        """
        Lists all connected GitHub repositories associated with the given project.

        Args:
            project_id: The UUID of the project.
        """
        try:
            repos = await prisma.projectrepository.find_many(where={"projectId": project_id})
            if not repos:
                return f"No GitHub repositories connected to project '{project_id}'."

            lines = [
                f"- Repository '{r.fullName}' [ID: {r.id}]: Default Branch={r.defaultBranch}, Owner={r.owner}, Repo={r.name}, Installed={r.isInstalled}"
                for r in repos
            ]
            return f"Repositories for Project '{project_id}':\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Error listing project repositories: {e}")
            return f"Error listing project repositories: {str(e)}"

    @mcp.tool()
    async def create_project_requirement(
        project_id: str,
        title: str,
        description: str,
        type: str = "FUNCTIONAL",
        priority: str = "P1"
    ) -> str:
        """
        Creates a new project requirement in PostgreSQL and synchronizes it into the Neo4j knowledge graph.

        Args:
            project_id: The UUID of the active project.
            title: Title of the requirement.
            description: Detailed technical specification and acceptance criteria.
            type: 'FUNCTIONAL' or 'NON_FUNCTIONAL' (default 'FUNCTIONAL').
            priority: Priority level: 'P0', 'P1', or 'P2' (default 'P1').
        """
        logger.info(f"[mcp: create_project_requirement] Creating requirement '{title}' for project '{project_id}'")
        try:
            type_norm = type.upper().replace("-", "_").strip()
            type_enum = RequirementType.NON_FUNCTIONAL if "NON" in type_norm else RequirementType.FUNCTIONAL

            priority_norm = priority.upper().strip()
            priority_map = {"P0": RequirementPriority.P0, "P1": RequirementPriority.P1, "P2": RequirementPriority.P2}
            priority_enum = priority_map.get(priority_norm, RequirementPriority.P1)

            req = await prisma.requirement.create(
                data={
                    "projectId": project_id,
                    "title": title.strip(),
                    "description": description.strip(),
                    "type": type_enum,
                    "priority": priority_enum,
                    "status": RequirementStatus.APPROVED,
                }
            )

            # Sync node and relation to Neo4j
            try:
                query = """
                MERGE (p:Project {id: $project_id})
                MERGE (r:Requirement {id: $req_id})
                SET r.title = $title, r.type = $type, r.priority = $priority, r.status = $status
                MERGE (p)-[:HAS_REQUIREMENT]->(r)
                """
                await neo4j_graph_store.execute_query(
                    query,
                    {
                        "project_id": project_id,
                        "req_id": req.id,
                        "title": req.title,
                        "type": str(req.type),
                        "priority": str(req.priority),
                        "status": str(req.status),
                    }
                )
            except Exception as graph_err:
                logger.warning(f"Neo4j sync for requirement '{req.id}' deferred or failed: {graph_err}")

            return f"Successfully created Requirement '{req.title}' [ID: {req.id}] with Type '{req.type}' and Priority '{req.priority}'."
        except Exception as e:
            logger.error(f"Error creating requirement: {e}")
            return f"Error creating requirement: {str(e)}"

    @mcp.tool()
    async def update_requirement_status(
        requirement_id: str,
        status: str
    ) -> str:
        """
        Updates the lifecycle status of a project requirement.

        Args:
            requirement_id: UUID of the requirement.
            status: Target status: 'DRAFT', 'SUGGESTION', 'APPROVED', or 'REJECTED'.
        """
        logger.info(f"[mcp: update_requirement_status] Updating requirement '{requirement_id}' status to '{status}'")
        try:
            status_map = {
                "DRAFT": RequirementStatus.DRAFT,
                "SUGGESTION": RequirementStatus.SUGGESTION,
                "APPROVED": RequirementStatus.APPROVED,
                "REJECTED": RequirementStatus.REJECTED,
            }
            norm = status.upper().replace(" ", "_").strip()
            if norm not in status_map:
                valid = ", ".join(status_map.keys())
                return f"Error: Invalid status '{status}'. Must be one of: {valid}"

            req = await prisma.requirement.update(
                where={"id": requirement_id},
                data={"status": status_map[norm]}
            )
            if not req:
                return f"Error: Requirement '{requirement_id}' not found."

            # Update Neo4j node status
            try:
                await neo4j_graph_store.execute_query(
                    "MATCH (r:Requirement {id: $req_id}) SET r.status = $status",
                    {"req_id": requirement_id, "status": norm}
                )
            except Exception as graph_err:
                logger.warning(f"Neo4j status update for requirement '{requirement_id}' bypassed: {graph_err}")

            return f"Requirement '{req.title}' [ID: {req.id}] status successfully updated to '{req.status}'."
        except Exception as e:
            logger.error(f"Error updating requirement status: {e}")
            return f"Error updating requirement status: {str(e)}"

    @mcp.tool()
    async def get_task_details(task_id: str) -> str:
        """
        Retrieves full details for a developer task including linked requirement and project.

        Args:
            task_id: UUID of the task.
        """
        logger.info(f"[mcp: get_task_details] Fetching task '{task_id}'")
        try:
            task = await prisma.task.find_unique(
                where={"id": task_id},
                include={
                    "requirement": True,
                    "project": True,
                }
            )
            if not task:
                return f"Error: Task '{task_id}' not found."

            req_info = (
                f"'{task.requirement.title}' [ID: {task.requirement.id}]"
                if task.requirement
                else "None (Unassigned)"
            )
            proj_name = task.project.name if task.project else task.projectId

            return (
                f"Task: '{task.title}' [ID: {task.id}]\n"
                f"- Project: {proj_name}\n"
                f"- Status: {task.status}\n"
                f"- Priority: {task.priority}\n"
                f"- Source: {task.source}\n"
                f"- Approval: {task.approvalStatus}\n"
                f"- Linked Requirement: {req_info}\n"
                f"- Description:\n{task.description}\n"
                f"- Created: {task.createdAt.isoformat()}"
            )
        except Exception as e:
            logger.error(f"Error fetching task details: {e}")
            return f"Error fetching task details: {str(e)}"

    @mcp.tool()
    async def assign_task_to_requirement(
        task_id: str,
        requirement_id: str
    ) -> str:
        """
        Assigns a developer task to a requirement, ensuring implementation traceability.

        Args:
            task_id: UUID of the task.
            requirement_id: UUID of the requirement this task implements.
        """
        logger.info(f"[mcp: assign_task_to_requirement] Assigning task '{task_id}' to requirement '{requirement_id}'")
        try:
            task = await prisma.task.find_unique(where={"id": task_id})
            if not task:
                return f"Error: Task '{task_id}' not found."

            req = await prisma.requirement.find_unique(where={"id": requirement_id})
            if not req:
                return f"Error: Requirement '{requirement_id}' not found."

            updated_task = await prisma.task.update(
                where={"id": task_id},
                data={"requirementId": requirement_id}
            )

            # Sync relationship to Neo4j
            try:
                query = """
                MATCH (t:Task {id: $task_id}), (r:Requirement {id: $req_id})
                MERGE (t)-[:IMPLEMENTS]->(r)
                """
                await neo4j_graph_store.execute_query(
                    query,
                    {"task_id": task_id, "req_id": requirement_id}
                )
            except Exception as graph_err:
                logger.warning(f"Neo4j relation link bypassed: {graph_err}")

            return f"Successfully assigned Task '{updated_task.title}' [ID: {task_id}] to Requirement '{req.title}' [ID: {requirement_id}]."
        except Exception as e:
            logger.error(f"Error assigning task to requirement: {e}")
            return f"Error assigning task to requirement: {str(e)}"

    @mcp.tool()
    async def get_conflict_details(conflict_id: str) -> str:
        """
        Retrieves full details of a specific requirement conflict.

        Args:
            conflict_id: UUID of the requirement conflict.
        """
        logger.info(f"[mcp: get_conflict_details] Fetching conflict '{conflict_id}'")
        try:
            c = await prisma.requirementconflict.find_unique(
                where={"id": conflict_id},
                include={
                    "requirement": True,
                    "conflictingRequirement": True,
                    "project": True,
                }
            )
            if not c:
                return f"Error: Conflict '{conflict_id}' not found."

            req_a = c.requirement
            req_b = c.conflictingRequirement
            req_a_str = f"'{req_a.title}' [ID: {req_a.id}, Type: {req_a.type}, Priority: {req_a.priority}]:\n  {req_a.description}" if req_a else "Unknown"
            req_b_str = f"'{req_b.title}' [ID: {req_b.id}, Type: {req_b.type}, Priority: {req_b.priority}]:\n  {req_b.description}" if req_b else "Unknown"

            return (
                f"Requirement Conflict [ID: {c.id}]:\n"
                f"- Project: {c.project.name if c.project else c.projectId}\n"
                f"- Status: {c.status}\n"
                f"- Severity: {c.severity}\n"
                f"- Description: {c.description}\n"
                f"- AI Recommendation: {c.aiRecommendation or 'No recommendation recorded yet'}\n"
                f"- Requirement A:\n  {req_a_str}\n"
                f"- Requirement B (Conflicting):\n  {req_b_str}\n"
                f"- Created: {c.createdAt.isoformat()}"
            )
        except Exception as e:
            logger.error(f"Error fetching conflict details: {e}")
            return f"Error fetching conflict details: {str(e)}"

    @mcp.tool()
    async def apply_ai_suggestion(suggestion_id: str) -> str:
        """
        Accepts and applies an AI suggestion, converting it into a project entity.

        Args:
            suggestion_id: UUID of the AI suggestion to accept.
        """
        logger.info(f"[mcp: apply_ai_suggestion] Applying suggestion '{suggestion_id}'")
        try:
            sugg = await prisma.aisuggestion.find_unique(where={"id": suggestion_id})
            if not sugg:
                return f"Error: AI Suggestion '{suggestion_id}' not found."

            # Mark suggestion accepted
            await prisma.aisuggestion.update(
                where={"id": suggestion_id},
                data={"status": SuggestionStatus.ACCEPTED}
            )

            content = sugg.content if isinstance(sugg.content, dict) else {}
            title = content.get("title", f"Suggested {sugg.type}")
            description = content.get("description", str(content))

            created_entity_str = ""
            s_type = sugg.type.lower()
            if "requirement" in s_type:
                new_req = await prisma.requirement.create(
                    data={
                        "projectId": sugg.projectId,
                        "title": title,
                        "description": description,
                        "type": RequirementType.FUNCTIONAL,
                        "priority": RequirementPriority.P1,
                        "status": RequirementStatus.APPROVED,
                    }
                )
                created_entity_str = f" Created Requirement '{new_req.title}' [ID: {new_req.id}]."
            elif "task" in s_type:
                new_task = await prisma.task.create(
                    data={
                        "projectId": sugg.projectId,
                        "title": title,
                        "description": description,
                        "priority": TaskPriority.P2,
                        "status": TaskStatus.TODO,
                        "source": TaskSource.SYSTEM,
                        "approvalStatus": TaskApprovalStatus.APPROVED,
                    }
                )
                created_entity_str = f" Created Task '{new_task.title}' [ID: {new_task.id}]."

            return f"AI Suggestion [ID: {suggestion_id}] successfully accepted.{created_entity_str}"
        except Exception as e:
            logger.error(f"Error applying AI suggestion: {e}")
            return f"Error applying AI suggestion: {str(e)}"

    @mcp.tool()
    async def reject_ai_suggestion(
        suggestion_id: str,
        reason: Optional[str] = None
    ) -> str:
        """
        Rejects and dismisses an AI suggestion.

        Args:
            suggestion_id: UUID of the AI suggestion to reject.
            reason: Optional explanation for rejection.
        """
        logger.info(f"[mcp: reject_ai_suggestion] Rejecting suggestion '{suggestion_id}'")
        try:
            sugg = await prisma.aisuggestion.find_unique(where={"id": suggestion_id})
            if not sugg:
                return f"Error: AI Suggestion '{suggestion_id}' not found."

            await prisma.aisuggestion.update(
                where={"id": suggestion_id},
                data={"status": SuggestionStatus.REJECTED}
            )
            reason_str = f" Reason: {reason}" if reason else ""
            return f"AI Suggestion [ID: {suggestion_id}] successfully rejected.{reason_str}"
        except Exception as e:
            logger.error(f"Error rejecting AI suggestion: {e}")
            return f"Error rejecting AI suggestion: {str(e)}"

    @mcp.tool()
    async def get_document_metadata(document_id: str) -> str:
        """
        Retrieves metadata and status for a specific uploaded document.

        Args:
            document_id: UUID of the document.
        """
        logger.info(f"[mcp: get_document_metadata] Fetching metadata for document '{document_id}'")
        try:
            doc = await prisma.document.find_unique(
                where={"id": document_id},
                include={"project": True}
            )
            if not doc:
                return f"Error: Document '{document_id}' not found."

            proj_name = doc.project.name if doc.project else doc.projectId
            return (
                f"Document: '{doc.name}' [ID: {doc.id}]\n"
                f"- Project: {proj_name}\n"
                f"- Status: {doc.status}\n"
                f"- File Type: {doc.fileType}\n"
                f"- Size: {doc.sizeBytes} bytes\n"
                f"- File Path: {doc.filePath}\n"
                f"- Created: {doc.createdAt.isoformat()}\n"
                f"- Updated: {doc.updatedAt.isoformat()}"
            )
        except Exception as e:
            logger.error(f"Error fetching document metadata: {e}")
            return f"Error fetching document metadata: {str(e)}"


