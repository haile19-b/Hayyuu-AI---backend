"""LangGraph Long-Term Episodic Memory Consolidator.

Extracts enduring facts, user preferences, and architectural decisions
at the conclusion of conversational milestones using Google GenAI SDK
with native structured outputs (response_schema=MemoryConsolidationOutput),
persisting them across LangGraph AsyncPostgresStore and Prisma MemoryRecord.
"""

from __future__ import annotations
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from google.genai import types

from app.core.config import genAI
from app.core.env import settings
from app.core.database import prisma, get_memory_store
from app.agents.supervisor.metamo.shared import clamp_to_unit_interval
from prisma.enums import MemoryType

logger = logging.getLogger("uvicorn.error")


class MemoryInsightType(str, Enum):
    """Classification taxonomy for extracted memory insights."""
    PREFERENCE = "PREFERENCE"
    DECISION = "DECISION"
    FACT = "FACT"
    RULE = "RULE"


class MemoryItem(BaseModel):
    """A durable, extracted memory insight."""
    key: str = Field(
        ...,
        description="Hierarchical slug identifier, e.g. 'architecture/database', 'preferences/coding_style', 'constraints/security'"
    )
    type: MemoryInsightType = Field(
        default=MemoryInsightType.DECISION,
        description="Classification: 'PREFERENCE' | 'DECISION' | 'FACT' | 'RULE'"
    )
    content: str = Field(
        ...,
        description="Concise, self-contained single-sentence summary of the decision, fact, preference, or rule"
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Continuous importance rating from 0.0 (low importance) to 1.0 (mission-critical architectural mandate)"
    )


class MemoryConsolidationOutput(BaseModel):
    """Native structured output container for long-term memory extraction."""
    memories: List[MemoryItem] = Field(
        default_factory=list,
        description="List of durable, high-value extracted insights from the conversation"
    )


def _heuristic_fallback_extraction(messages: List[Dict[str, Any]]) -> List[MemoryItem]:
    """Deterministic fallback for local offline testing or when LLM API is unavailable."""
    extracted: List[MemoryItem] = []
    
    for m in messages:
        content = m.get("content", "")
        c_lower = content.lower()

        # Decision patterns
        if "decided to use" in c_lower or "we will use" in c_lower or "chosen" in c_lower:
            match = re.search(r"(?:decided to use|we will use|chosen)\s+([a-zA-Z0-9_\-\s]+)", content, re.IGNORECASE)
            target = match.group(1).strip().split(".")[0][:40] if match else "technology_stack"
            key = f"architecture/{re.sub(r'[^a-zA-Z0-9_]', '_', target.lower())}"
            extracted.append(MemoryItem(
                key=key,
                type=MemoryInsightType.DECISION,
                content=content[:200].strip(),
                importance=0.85
            ))

        # Preference patterns
        elif "prefer" in c_lower or "always use" in c_lower or "coding style" in c_lower:
            extracted.append(MemoryItem(
                key="preferences/user_style",
                type=MemoryInsightType.PREFERENCE,
                content=content[:200].strip(),
                importance=0.75
            ))

        # Rule / Constraint patterns
        elif "must ensure" in c_lower or "rule:" in c_lower or "policy" in c_lower or "never" in c_lower:
            extracted.append(MemoryItem(
                key="constraints/project_rule",
                type=MemoryInsightType.RULE,
                content=content[:200].strip(),
                importance=0.90
            ))

    return extracted


async def consolidate_thread_memories(
    user_id: str,
    project_id: Optional[str],
    thread_id: str,
    messages: List[Dict[str, Any]],
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extracts durable episodic insights using Google GenAI SDK native Structured Outputs.
    Synchronizes storage across LangGraph AsyncPostgresStore and Prisma MemoryRecord.
    """
    if len(messages) < 2:
        return []

    # Format transcript snippet for extraction
    transcript_lines = []
    for m in messages[-12:]:
        role = m.get("role", "USER")
        content = m.get("content", "").strip()
        transcript_lines.append(f"{role}: {content[:350]}")
    transcript_text = "\n".join(transcript_lines)

    extracted_items: List[MemoryItem] = []
    model = model_name or (settings.GEMINI_MODELS[0] if settings.GEMINI_MODELS else "gemini-3.8-flash")

    system_instruction = (
        "You are an episodic memory consolidation system for an AI software engineering assistant.\n"
        "Analyze the conversational transcript and extract durable, high-value insights:\n"
        "1. User preferences (coding conventions, communication style, tools).\n"
        "2. Architectural decisions (frameworks, database design, API styles, auth).\n"
        "3. Domain facts or constraints (compliance, project targets, performance limits).\n"
        "If there are no meaningful or durable insights to preserve, return an empty memories list."
    )

    try:
        response = await genAI.aio.models.generate_content(
            model=model,
            contents=[
                f"{system_instruction}\n\nConversation Transcript:\n{transcript_text}"
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MemoryConsolidationOutput,
                temperature=0.0,
            ),
        )

        if response.text:
            result = MemoryConsolidationOutput.model_validate_json(response.text)
            extracted_items = result.memories
    except Exception as e:
        logger.warning(f"Consolidation LLM extraction fell back to heuristic: {e}")
        extracted_items = _heuristic_fallback_extraction(messages)

    if not extracted_items:
        return []

    saved_records = []
    store = None
    try:
        store = await get_memory_store()
    except Exception as store_err:
        logger.debug(f"LangGraph Store unavailable: {store_err}")

    # Map insight types to Prisma MemoryType
    type_mapping = {
        MemoryInsightType.PREFERENCE: MemoryType.EPISODIC,
        MemoryInsightType.DECISION: MemoryType.EPISODIC,
        MemoryInsightType.FACT: MemoryType.SEMANTIC,
        MemoryInsightType.RULE: MemoryType.PROCEDURAL,
    }

    for item in extracted_items:
        key = item.key.strip()
        content = item.content.strip()
        if not key or not content:
            continue

        mem_type = type_mapping.get(item.type, MemoryType.EPISODIC)
        importance = clamp_to_unit_interval(item.importance)

        # 1. Save to LangGraph Postgres Store
        if store is not None:
            try:
                namespace = ("memories", user_id, project_id) if project_id else ("memories", user_id, "global")
                await store.aput(
                    namespace,
                    key=key,
                    value={
                        "content": content,
                        "type": mem_type.value,
                        "importance": importance,
                        "project_id": project_id,
                        "thread_id": thread_id,
                        "updated_at": datetime.now().isoformat()
                    }
                )
            except Exception as store_err:
                logger.debug(f"Could not persist to LangGraph store: {store_err}")

        # 2. Persist to Prisma MemoryRecord
        try:
            where_clause: Dict[str, Any] = {"userId": user_id, "key": key}
            if project_id:
                where_clause["projectId"] = project_id

            existing = await prisma.memoryrecord.find_first(where=where_clause)
            if existing:
                updated = await prisma.memoryrecord.update(
                    where={"id": existing.id},
                    data={
                        "content": content,
                        "type": mem_type,
                        "importance": importance,
                        "projectId": project_id,
                        "lastAccessed": datetime.now()
                    }
                )
                saved_records.append({"id": updated.id, "key": key, "action": "updated", "type": mem_type.value})
            else:
                created = await prisma.memoryrecord.create(
                    data={
                        "userId": user_id,
                        "projectId": project_id,
                        "key": key,
                        "type": mem_type,
                        "content": content,
                        "importance": importance
                    }
                )
                saved_records.append({"id": created.id, "key": key, "action": "created", "type": mem_type.value})
        except Exception as prisma_err:
            logger.warning(f"Failed to upsert Prisma memory record: {prisma_err}")

    logger.info(f"Consolidated {len(saved_records)} long-term memories for user '{user_id}'")
    return saved_records
