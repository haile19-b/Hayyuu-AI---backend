import logging
from typing import Dict, Any, List, Optional
from app.core.database import prisma
from app.agents.memory.retriever import format_memories_for_prompt

logger = logging.getLogger("uvicorn.error")


def estimate_tokens(text: str) -> int:
    """Estimates token count using standard 4-characters per token heuristic."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def format_history_for_genai(
    tier1_messages: List[Dict[str, Any]],
    current_prompt: str
) -> List[Any]:
    """
    Transforms chronological chat history into Gemini-compatible types.Content objects.
    Ensures alternating user/model roles and terminates with the active user prompt.
    """
    from google.genai import types
    contents: List[types.Content] = []

    history = [m for m in tier1_messages if isinstance(m, dict)]
    # If the newest message in history is the just-saved current prompt, pop it to cleanly place as final turn
    if history and history[-1].get("role", "").upper() == "USER" and history[-1].get("content", "").strip() == current_prompt.strip():
        history.pop()

    last_role = None
    for msg in history:
        raw_role = msg.get("role", "USER").upper()
        genai_role = "user" if raw_role == "USER" else "model"
        text = (msg.get("content") or "").strip()
        if not text:
            continue

        # Merge consecutive identical roles to adhere to strict GenAI alternating constraints
        if genai_role == last_role and contents:
            prev_parts = contents[-1].parts
            if prev_parts:
                prev_text = getattr(prev_parts[0], "text", "") or ""
                contents[-1].parts = [types.Part.from_text(text=f"{prev_text}\n\n{text}")]
            continue

        contents.append(
            types.Content(
                role=genai_role,
                parts=[types.Part.from_text(text=text)]
            )
        )
        last_role = genai_role

    # Append current user prompt as the final turn
    if last_role == "user" and contents:
        prev_parts = contents[-1].parts
        if prev_parts:
            prev_text = getattr(prev_parts[0], "text", "") or ""
            contents[-1].parts = [types.Part.from_text(text=f"{prev_text}\n\n{current_prompt}")]
    else:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=current_prompt)]
            )
        )

    return contents


async def assemble_context_with_budget(
    thread_id: str,
    project_id: str,
    user_query: str,
    action: Optional[str] = None,
    intent_type: Optional[str] = None,
    total_budget_ceiling: int = 3500,
    tier1_tokens: int = 1500,
    tier2_tokens: int = 1000,
    kg_tokens: int = 1000,
    semantic_search: bool = True,
    memories: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Assembles multi-tier conversation and domain context respecting strict token budgets
    and intent-gated resource injection.

    Tier 1 (Base History): Recent chronological message turns up to tier1_tokens.
    Tier 2 (Semantic Recall): Semantically relevant historical turns (excluding Tier 1) up to tier2_tokens.
    KG Context: Base project anchor (<250 tokens) + intent-gated domain resources up to kg_tokens.
    Ceiling: Enforces total_budget_ceiling via priority-based trimming.
    """
    tier1_messages: List[Dict[str, Any]] = []
    tier2_messages: List[Dict[str, Any]] = []
    kg_context_parts: List[str] = []

    # -------------------------------------------------------------------------
    # 1. Tier 1: Recent Chat History (Chronological Window)
    # -------------------------------------------------------------------------
    tier1_token_count = 0
    tier1_ids = set()

    try:
        # Fetch up to 40 recent messages
        recent_msgs = await prisma.chatmessage.find_many(
            where={"threadId": thread_id},
            order={"createdAt": "desc"},
            take=40
        )
        recent_msgs.reverse()

        # Add messages newest backwards until tier1_tokens budget is filled
        selected_recent = []
        for msg in reversed(recent_msgs):
            content_tokens = estimate_tokens(msg.content)
            if tier1_token_count + content_tokens <= tier1_tokens or not selected_recent:
                selected_recent.append(msg)
                tier1_token_count += content_tokens
                tier1_ids.add(msg.id)
            else:
                break

        selected_recent.reverse()
        for msg in selected_recent:
            role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role).upper()
            tier1_messages.append({
                "id": msg.id,
                "role": role_val,
                "content": msg.content,
                "action": msg.cognitiveAction,
                "style": msg.styleModifier,
                "created_at": msg.createdAt.isoformat()
            })
    except Exception as e:
        logger.warning(f"Error fetching Tier 1 chat history: {e}")

    # -------------------------------------------------------------------------
    # 2. Tier 2: Semantically Relevant Chat Turns (On-Demand Raw Recall)
    # -------------------------------------------------------------------------
    tier2_token_count = 0
    if user_query and user_query.strip():
        try:
            candidates: List[Dict[str, Any]] = []
            if semantic_search:
                try:
                    from app.infrastructure.ai.gemini_embedder import GeminiEmbedder
                    embedder = GeminiEmbedder()
                    query_vec = embedder.embed_text(user_query, user_query=True)
                    vec_literal = "[" + ",".join(str(x) for x in query_vec) + "]"

                    raw_res = await prisma.query_raw(
                        f"""
                        SELECT id, role, content, "cognitiveAction", "styleModifier", "createdAt",
                               (1 - (embedding <=> '{vec_literal}'::vector)) AS similarity
                        FROM chat_messages
                        WHERE "threadId" = $1 AND embedding IS NOT NULL
                        ORDER BY embedding <=> '{vec_literal}'::vector ASC
                        LIMIT 10;
                        """,
                        thread_id
                    )
                    if raw_res:
                        for r in raw_res:
                            candidates.append({
                                "id": r.get("id"),
                                "role": r.get("role"),
                                "content": r.get("content") or "",
                                "action": r.get("cognitiveAction"),
                                "style": r.get("styleModifier"),
                                "created_at": str(r.get("createdAt")),
                                "similarity": float(r.get("similarity") or 0.0)
                            })
                except Exception as embed_err:
                    logger.debug(f"Tier 2 semantic embedding lookup bypassed: {embed_err}")

            if not candidates:
                # Text-based fallback keyword matching
                words = [w for w in user_query.split() if len(w) > 3][:3]
                for word in words:
                    matches = await prisma.chatmessage.find_many(
                        where={
                            "threadId": thread_id,
                            "content": {"contains": word, "mode": "insensitive"}
                        },
                        order={"createdAt": "desc"},
                        take=5
                    )
                    for m in matches:
                        candidates.append({
                            "id": m.id,
                            "role": m.role.value if hasattr(m.role, "value") else str(m.role).upper(),
                            "content": m.content,
                            "action": m.cognitiveAction,
                            "style": m.styleModifier,
                            "created_at": m.createdAt.isoformat(),
                            "similarity": 0.5
                        })

            # Filter candidates: exclude turns already in Tier 1
            for cand in candidates:
                cand_id = cand["id"]
                if cand_id in tier1_ids:
                    continue
                c_tokens = estimate_tokens(cand["content"])
                if tier2_token_count + c_tokens <= tier2_tokens:
                    tier2_messages.append(cand)
                    tier2_token_count += c_tokens
                    tier1_ids.add(cand_id)
        except Exception as e:
            logger.warning(f"Error fetching Tier 2 search turns: {e}")

    # -------------------------------------------------------------------------
    # 3. Knowledge Graph & Project Entities (Intent-Gated Dynamic Context)
    # -------------------------------------------------------------------------
    kg_token_count = 0
    q_lower = user_query.lower() if user_query else ""

    try:
        # Base Anchor: Project Overview, Connected Repositories & Documents - Always Included
        project = await prisma.project.find_unique(
            where={"id": project_id},
            include={
                "repositories": True,
                "documents": True,
            }
        )
        if project:
            anchor_lines = [
                f"[Project Anchor: {project.name}]",
                f"Status: {project.status} | Description: {project.description or 'N/A'}"
            ]
            if project.repositories:
                repos_summary = ", ".join([
                    f"{r.fullName} (URL: {r.htmlUrl}, Branch: {r.defaultBranch}, Active: {r.isActive})"
                    for r in project.repositories
                ])
                anchor_lines.append(f"Connected GitHub Repositories: {repos_summary}")
            else:
                anchor_lines.append("Connected GitHub Repositories: None")

            if project.documents:
                docs_summary = ", ".join([
                    f"'{d.name}' ({d.fileType}, Status: {d.status})"
                    for d in project.documents
                ])
                anchor_lines.append(f"Uploaded Documents: {docs_summary}")
            else:
                anchor_lines.append("Uploaded Documents: None")

            base_anchor = "\n".join(anchor_lines)
            base_tokens = estimate_tokens(base_anchor)
            kg_context_parts.append(base_anchor)
            kg_token_count += base_tokens

        # Intent Gating: Repositories & Codebase (Included for repo/github/git/code queries)
        should_include_repos = (
            "repo" in q_lower
            or "github" in q_lower
            or "git" in q_lower
            or "codebase" in q_lower
            or "branch" in q_lower
            or "commit" in q_lower
            or "pull request" in q_lower
            or "pr" in q_lower
        )
        if should_include_repos:
            repos = await prisma.projectrepository.find_many(
                where={"projectId": project_id}
            )
            if repos:
                repo_lines = ["[Connected GitHub Repositories Details]"]
                for r in repos:
                    repo_lines.append(
                        f"- Repository [{r.fullName}]:\n"
                        f"  URL: {r.htmlUrl}\n"
                        f"  Owner: {r.owner}\n"
                        f"  Name: {r.name}\n"
                        f"  Default Branch: {r.defaultBranch}\n"
                        f"  Active Status: {'Active' if r.isActive else 'Inactive'}"
                    )
                r_text = "\n".join(repo_lines)
                r_tokens = estimate_tokens(r_text)
                if kg_token_count + r_tokens <= kg_tokens:
                    kg_context_parts.append(r_text)
                    kg_token_count += r_tokens

        # Intent Gating: Uploaded Documents & Specifications
        should_include_docs = (
            "document" in q_lower
            or "doc" in q_lower
            or "file" in q_lower
            or "pdf" in q_lower
            or "spec" in q_lower
            or "srs" in q_lower
        )
        if should_include_docs:
            docs = await prisma.document.find_many(
                where={"projectId": project_id},
                take=10,
                order={"createdAt": "desc"}
            )
            if docs:
                doc_lines = ["[Uploaded Project Documents Index]"]
                for d in docs:
                    doc_lines.append(f"- Document [{d.id[:8]}] '{d.name}': Status={d.status}, Type={d.fileType}, Size={d.sizeBytes} bytes")
                d_text = "\n".join(doc_lines)
                d_tokens = estimate_tokens(d_text)
                if kg_token_count + d_tokens <= kg_tokens:
                    kg_context_parts.append(d_text)
                    kg_token_count += d_tokens

        # Intent Gating: Active Conflicts (Included for act_verify or conflict queries)
        should_include_conflicts = (
            action == "act_verify" 
            or "conflict" in q_lower 
            or "verify" in q_lower 
            or "contradict" in q_lower
        )
        if should_include_conflicts:
            conflicts = await prisma.requirementconflict.find_many(
                where={"projectId": project_id, "status": "ACTIVE"},
                take=5
            )
            if conflicts:
                conf_lines = ["[Active Requirement Conflicts]"]
                for c in conflicts:
                    conf_lines.append(f"- Conflict [{c.id[:8]}]: Severity={c.severity}, Description={c.description[:100]}")
                c_text = "\n".join(conf_lines)
                c_tokens = estimate_tokens(c_text)
                if kg_token_count + c_tokens <= kg_tokens:
                    kg_context_parts.append(c_text)
                    kg_token_count += c_tokens

        # Intent Gating: Tasks Breakdown (Included for act_decompose or task queries)
        should_include_tasks = (
            action == "act_decompose"
            or "task" in q_lower
            or "sprint" in q_lower
            or "plan" in q_lower
            or "todo" in q_lower
        )
        if should_include_tasks:
            tasks = await prisma.task.find_many(
                where={"projectId": project_id},
                take=6,
                order={"createdAt": "desc"}
            )
            if tasks:
                task_lines = ["[Active Tasks Index]"]
                for t in tasks:
                    task_lines.append(f"- Task [{t.id[:8]}] '{t.title}': Status={t.status}, Priority={t.priority}")
                t_text = "\n".join(task_lines)
                t_tokens = estimate_tokens(t_text)
                if kg_token_count + t_tokens <= kg_tokens:
                    kg_context_parts.append(t_text)
                    kg_token_count += t_tokens

        # Intent Gating: Critical Requirements (Included for act_search, act_synthesize or spec queries)
        should_include_reqs = (
            action in ["act_search", "act_synthesize"]
            or "requirement" in q_lower
            or "spec" in q_lower
            or "feature" in q_lower
            or not kg_context_parts
        )
        if should_include_reqs:
            reqs = await prisma.requirement.find_many(
                where={"projectId": project_id},
                take=6,
                order={"priority": "asc"}
            )
            if reqs:
                req_lines = ["[Core Requirements Index]"]
                for r in reqs:
                    req_lines.append(f"- Req [{r.id[:8]}] '{r.title}': Priority={r.priority}, Type={r.type}")
                r_text = "\n".join(req_lines)
                r_tokens = estimate_tokens(r_text)
                if kg_token_count + r_tokens <= kg_tokens:
                    kg_context_parts.append(r_text)
                    kg_token_count += r_tokens
    except Exception as e:
        logger.warning(f"Error fetching intent-gated domain context: {e}")

    # -------------------------------------------------------------------------
    # 4. Total Budget Ceiling Enforcement & Priority Trimming
    # -------------------------------------------------------------------------
    # Format long-term memories if provided
    memory_context = format_memories_for_prompt(memories) if memories else ""
    mem_tokens = estimate_tokens(memory_context)
    total_tokens = tier1_token_count + tier2_token_count + kg_token_count + mem_tokens

    # If exceeding ceiling, trim Tier 2 first, then KG context, preserving Tier 1
    if total_tokens > total_budget_ceiling:
        overflow = total_tokens - total_budget_ceiling
        # Trim from Tier 2
        while tier2_messages and overflow > 0:
            removed = tier2_messages.pop()
            r_tok = estimate_tokens(removed["content"])
            overflow -= r_tok
            tier2_token_count -= r_tok

        # If still overflowing, trim non-anchor KG context parts
        while len(kg_context_parts) > 1 and overflow > 0:
            removed_kg = kg_context_parts.pop()
            r_tok = estimate_tokens(removed_kg)
            overflow -= r_tok
            kg_token_count -= r_tok

        total_tokens = tier1_token_count + tier2_token_count + kg_token_count + mem_tokens

    return {
        "tier1_chat_history": tier1_messages,
        "tier2_relevant_turns": tier2_messages,
        "kg_context": "\n\n".join(kg_context_parts),
        "memory_context": memory_context,
        "total_estimated_tokens": total_tokens
    }
