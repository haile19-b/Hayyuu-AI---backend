"""LangGraph Long-Term Episodic Memory Consolidator.

Extracts enduring facts, user preferences, and architectural decisions
at the conclusion of conversational milestones using Google GenAI SDK
with native structured outputs (response_schema=MemoryConsolidationOutput),
persisting them across LangGraph AsyncPostgresStore and Prisma MemoryRecord.
"""

from __future__ import annotations
import re
import asyncio
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
    USER_PROFILE = "USER_PROFILE"
    ASSIGNMENT = "ASSIGNMENT"
    PREFERENCE = "PREFERENCE"
    DECISION = "DECISION"
    FACT = "FACT"
    RULE = "RULE"


class MemoryItem(BaseModel):
    """A durable, extracted memory insight."""
    key: str = Field(
        ...,
        description="Hierarchical slug identifier, e.g. 'profile/name', 'profile/role', 'architecture/database', 'constraints/security'"
    )
    type: MemoryInsightType = Field(
        default=MemoryInsightType.DECISION,
        description="Classification: 'USER_PROFILE' | 'ASSIGNMENT' | 'PREFERENCE' | 'DECISION' | 'FACT' | 'RULE'"
    )
    content: str = Field(
        ...,
        description="Concise, self-contained single-sentence summary of the decision, fact, preference, profile, or rule"
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Continuous importance rating from 0.0 (low importance) to 1.0 (mission-critical user/architectural mandate)"
    )


class MemoryConsolidationOutput(BaseModel):
    """Native structured output container for long-term memory extraction."""
    memories: List[MemoryItem] = Field(
        default_factory=list,
        description="List of durable, high-value extracted insights from the conversation"
    )


async def consolidate_thread_memories(
    user_id: str,
    project_id: Optional[str],
    thread_id: str,
    messages: List[Dict[str, Any]],
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extracts durable episodic insights strictly using Google GenAI SDK native Structured Outputs.
    Synchronizes storage across LangGraph AsyncPostgresStore and Prisma MemoryRecord.
    Never relies on regex or keyword heuristics.
    """
    if not messages:
        return []

    # Format transcript snippet for extraction
    transcript_lines = []
    for m in messages[-14:]:
        role = m.get("role", "USER")
        content = m.get("content", "").strip()
        transcript_lines.append(f"{role}: {content[:350]}")
    transcript_text = "\n".join(transcript_lines)

    from app.core.ai_cascade import generate_content_cascade, get_candidate_models

    system_instruction = (
        "You are an episodic and semantic memory consolidation engine for Hayyuu AI.\n"
        "Your task is to analyze the conversation transcript and extract durable, high-value insights about the user and the project.\n\n"
        "Categories to extract:\n"
        "1. USER_PROFILE: User identity, preferred name, title, role, skills, personal background (e.g. key: 'profile/name', 'profile/role').\n"
        "2. ASSIGNMENT: What project parts, modules, or tasks the user is assigned to or working on (e.g. key: 'profile/assignment', 'project/assigned_module').\n"
        "3. PREFERENCE: User preferences (coding conventions, framework choices, communication style, tools).\n"
        "4. DECISION: Architectural and technical decisions made for the project (database, APIs, auth, design patterns).\n"
        "5. FACT: Important facts or specifications about the project (requirements, repo structure, deadlines, team structure).\n"
        "6. RULE: Hard rules, constraints, policies, or invariants that must never be violated.\n\n"
        "Mandate: If the user states their name, role, module assignment, or important project details, ALWAYS extract them.\n"
        "If there are no meaningful or durable insights to preserve, return an empty memories list."
    )

    extracted_items: List[MemoryItem] = []
    try:
        response = await generate_content_cascade(
            contents=[f"{system_instruction}\n\nConversation Transcript:\n{transcript_text}"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MemoryConsolidationOutput,
                temperature=0.0,
            ),
            candidate_models=get_candidate_models(model_name),
            timeout_seconds=15.0,
        )
        if response and response.text:
            result = MemoryConsolidationOutput.model_validate_json(response.text)
            extracted_items = result.memories
    except Exception as e:
        logger.error(f"[Memory Consolidator] All cascade models failed for memory extraction: {e}")
        return []

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
        MemoryInsightType.USER_PROFILE: MemoryType.EPISODIC,
        MemoryInsightType.ASSIGNMENT: MemoryType.EPISODIC,
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

        # 1. Save to LangGraph Postgres Store across namespaces
        if store is not None:
            try:
                namespace = ("memories", user_id, project_id) if project_id else ("memories", user_id, "global")
                payload = {
                    "content": content,
                    "type": mem_type.value,
                    "importance": importance,
                    "project_id": project_id,
                    "thread_id": thread_id,
                    "updated_at": datetime.now().isoformat()
                }
                await store.aput(namespace, key=key, value=payload)
                if key.startswith("profile/"):
                    await store.aput(("memories", user_id), key=key, value=payload)
            except Exception as store_err:
                logger.debug(f"Could not persist to LangGraph store: {store_err}")

        # 2. Persist to Prisma MemoryRecord
        try:
            where_clause: Dict[str, Any] = {"userId": user_id, "key": key}
            if not key.startswith("profile/") and project_id:
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

            # 3. Synchronize valid user fullName to User table if extracted by LLM
            if key == "profile/name":
                try:
                    cleaned_name = content
                    for prefix in ("The user's name is", "User's name is", "User's preferred name is", "User is", "Name is"):
                        if cleaned_name.lower().startswith(prefix.lower()):
                            cleaned_name = cleaned_name[len(prefix):].strip()
                    cleaned_name = cleaned_name.rstrip(".,!?:")
                    if cleaned_name and len(cleaned_name) > 1:
                        await prisma.user.update(where={"id": user_id}, data={"fullName": cleaned_name})
                except Exception as u_err:
                    logger.debug(f"User fullName update deferred: {u_err}")

        except Exception as prisma_err:
            logger.warning(f"Failed to upsert Prisma memory record: {prisma_err}")

    logger.info(f"Consolidated {len(saved_records)} long-term memories for user '{user_id}'")
    return saved_records
