"""LangGraph Long-Term Episodic Memory Retriever.

Retrieves durable user preferences, architectural decisions, and domain insights
using LangGraph AsyncPostgresStore and Prisma MemoryRecord indexes.
Formats retrieved memories for hot-path prompt injection under <project_memory>.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.database import prisma, get_memory_store

logger = logging.getLogger("uvicorn.error")


def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved memory dictionaries into a prompt-injectable
    <project_memory> XML block.
    """
    if not memories:
        return ""

    lines = ["<project_memory>"]
    for m in memories:
        m_type = m.get("type", "FACT").upper()
        key = m.get("key", "unnamed")
        content = m.get("content", "").strip()
        importance = float(m.get("importance", 0.5))
        lines.append(f"- [{m_type}] {key}: {content} (importance: {importance:.2f})")
    lines.append("</project_memory>")
    return "\n".join(lines)


async def retrieve_memories(
    user_id: str,
    project_id: Optional[str] = None,
    query: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieves relevant long-term episodic memories for a user and optional project scope.
    Combines LangGraph Store namespace retrieval with Prisma MemoryRecord tracking.
    """
    memories: List[Dict[str, Any]] = []
    seen_keys = set()

    # 1. Attempt LangGraph Postgres Store retrieval across scopes
    try:
        store = await get_memory_store()
        if store is not None and query and query.strip():
            namespaces_to_search = []
            if project_id:
                namespaces_to_search.append(("memories", user_id, project_id))
            namespaces_to_search.append(("memories", user_id, "global"))
            namespaces_to_search.append(("memories", user_id))

            for ns in namespaces_to_search:
                try:
                    store_results = await store.asearch(ns, query=query.strip(), limit=limit)
                    for item in store_results:
                        if item.key in seen_keys:
                            continue
                        seen_keys.add(item.key)
                        val = item.value if hasattr(item, "value") else {}
                        memories.append({
                            "key": item.key,
                            "content": val.get("content", str(val)),
                            "type": val.get("type", "FACT"),
                            "importance": float(val.get("importance", 0.5)),
                            "source": "langgraph_store",
                        })
                except Exception as ns_err:
                    logger.debug(f"LangGraph Store search in namespace {ns} failed: {ns_err}")
    except Exception as store_err:
        logger.debug(f"LangGraph Store search bypassed or returned no matches: {store_err}")

    # 2. Relational fallback / complement from Prisma MemoryRecord
    try:
        where: Dict[str, Any] = {"userId": user_id}
        if project_id:
            where["OR"] = [{"projectId": project_id}, {"projectId": None}]
        if memory_type:
            s_type = memory_type.upper().strip()
            from prisma.enums import MemoryType
            if s_type in MemoryType.__members__:
                where["type"] = MemoryType[s_type]

        records = await prisma.memoryrecord.find_many(
            where=where,
            order={"importance": "desc"},
            take=limit * 2
        )

        for rec in records:
            if rec.key in seen_keys:
                continue
            seen_keys.add(rec.key)
            memories.append({
                "id": rec.id,
                "key": rec.key,
                "content": rec.content,
                "type": rec.type.value if hasattr(rec.type, "value") else str(rec.type),
                "importance": rec.importance,
                "access_count": rec.accessCount,
                "source": "memory_record",
            })
            # Increment access count asynchronously
            try:
                await prisma.memoryrecord.update(
                    where={"id": rec.id},
                    data={
                        "accessCount": rec.accessCount + 1,
                        "lastAccessed": datetime.now()
                    }
                )
            except Exception:
                pass

    except Exception as prisma_err:
        logger.warning(f"Error querying Prisma memory records: {prisma_err}")

    # Sort descending by importance
    memories.sort(key=lambda m: float(m.get("importance", 0.5)), reverse=True)
    return memories[:limit]
