"""User Memory Management REST Endpoints.

Provides direct CRUD operations for project and user episodic/semantic memories,
synchronizing updates across Prisma MemoryRecord and LangGraph AsyncPostgresStore.
"""

from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from fastapi import APIRouter, Depends, Response, status, Query
from app.core.database import prisma, get_memory_store
from app.middleware.auth_middleware import auth_middleware
from app.domain.entities.auth.auth_schema import ApiResponse
from app.agents.supervisor.metamo.shared import clamp_to_unit_interval
from prisma.enums import MemoryType

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/projects", tags=["Memory Management"])


class MemoryItemResponse(BaseModel):
    id: str
    userId: str
    projectId: Optional[str]
    type: str
    key: str
    content: str
    importance: float
    accessCount: int
    lastAccessed: str
    createdAt: str
    updatedAt: str


class CreateMemoryRequest(BaseModel):
    key: str = Field(..., description="Hierarchical key, e.g. 'architecture/database'")
    content: str = Field(..., description="Text summary of the insight, decision, or preference")
    type: Optional[str] = Field("EPISODIC", description="SEMANTIC, EPISODIC, or PROCEDURAL")
    importance: Optional[float] = Field(0.5, description="Float importance between 0.0 and 1.0")


class UpdateMemoryRequest(BaseModel):
    key: Optional[str] = None
    content: Optional[str] = None
    type: Optional[str] = None
    importance: Optional[float] = None


@router.get("/{project_id}/memories", response_model=ApiResponse[List[MemoryItemResponse]])
async def list_project_memories(
    project_id: str,
    response: Response,
    memory_type: Optional[str] = Query(None, alias="type"),
    search: Optional[str] = None,
    limit: int = 50,
    credentials = Depends(auth_middleware)
):
    """Lists persistent memories for the project and user with optional filtering."""
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        where: Dict[str, Any] = {
            "userId": user_id,
            "OR": [{"projectId": project_id}, {"projectId": None}]
        }

        if memory_type:
            s_type = memory_type.upper().strip()
            if s_type in MemoryType.__members__:
                where["type"] = MemoryType[s_type]

        if search:
            q = search.strip()
            where["AND"] = [
                {"OR": [
                    {"key": {"contains": q, "mode": "insensitive"}},
                    {"content": {"contains": q, "mode": "insensitive"}}
                ]}
            ]

        records = await prisma.memoryrecord.find_many(
            where=where,
            order={"importance": "desc"},
            take=min(limit, 100)
        )

        items = [
            MemoryItemResponse(
                id=r.id,
                userId=r.userId,
                projectId=r.projectId,
                type=r.type.value if hasattr(r.type, "value") else str(r.type),
                key=r.key,
                content=r.content,
                importance=r.importance,
                accessCount=r.accessCount,
                lastAccessed=r.lastAccessed.isoformat(),
                createdAt=r.createdAt.isoformat(),
                updatedAt=r.updatedAt.isoformat()
            )
            for r in records
        ]

        return {"success": True, "response": items}
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.post("/{project_id}/memories", response_model=ApiResponse[MemoryItemResponse])
async def create_memory(
    project_id: str,
    body: CreateMemoryRequest,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """Creates a new memory record and stores it in both Prisma and LangGraph Store."""
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        m_type = MemoryType.EPISODIC
        raw_type = body.type.upper().strip()
        if raw_type in MemoryType.__members__:
            m_type = MemoryType[raw_type]

        # 1. Create in Prisma
        created = await prisma.memoryrecord.create(
            data={
                "userId": user_id,
                "projectId": project_id,
                "key": body.key.strip(),
                "content": body.content.strip(),
                "type": m_type,
                "importance": clamp_to_unit_interval(body.importance if body.importance is not None else 0.5),
            }
        )

        # 2. Sync to LangGraph Store
        try:
            store = await get_memory_store()
            if store is not None:
                payload = {
                    "content": created.content,
                    "type": m_type.value,
                    "importance": created.importance,
                    "project_id": project_id,
                    "updated_at": datetime.now().isoformat()
                }
                await store.aput(("memories", user_id, project_id), key=created.key, value=payload)
                await store.aput(("memories", user_id), key=created.key, value=payload)
        except Exception as store_err:
            logger.debug(f"LangGraph store sync deferred: {store_err}")

        item = MemoryItemResponse(
            id=created.id,
            userId=created.userId,
            projectId=created.projectId,
            type=created.type.value if hasattr(created.type, "value") else str(created.type),
            key=created.key,
            content=created.content,
            importance=created.importance,
            accessCount=created.accessCount,
            lastAccessed=created.lastAccessed.isoformat(),
            createdAt=created.createdAt.isoformat(),
            updatedAt=created.updatedAt.isoformat()
        )
        response.status_code = status.HTTP_201_CREATED
        return {"success": True, "response": item}
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.put("/{project_id}/memories/{memory_id}", response_model=ApiResponse[MemoryItemResponse])
async def update_memory(
    project_id: str,
    memory_id: str,
    body: UpdateMemoryRequest,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """Updates an existing memory record in Prisma and LangGraph Store."""
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        existing = await prisma.memoryrecord.find_unique(where={"id": memory_id})
        if not existing or existing.userId != user_id:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"success": False, "error": "Memory record not found"}

        data: Dict[str, Any] = {}
        if body.key:
            data["key"] = body.key.strip()
        if body.content:
            data["content"] = body.content.strip()
        if body.importance is not None:
            data["importance"] = clamp_to_unit_interval(body.importance)
        if body.type:
            raw_type = body.type.upper().strip()
            if raw_type in MemoryType.__members__:
                data["type"] = MemoryType[raw_type]

        updated = await prisma.memoryrecord.update(
            where={"id": memory_id},
            data=data
        )

        # Sync to LangGraph Store
        try:
            store = await get_memory_store()
            if store is not None:
                payload = {
                    "content": updated.content,
                    "type": updated.type.value if hasattr(updated.type, "value") else str(updated.type),
                    "importance": updated.importance,
                    "project_id": project_id,
                    "updated_at": datetime.now().isoformat()
                }
                await store.aput(("memories", user_id, project_id), key=updated.key, value=payload)
                await store.aput(("memories", user_id), key=updated.key, value=payload)
        except Exception as store_err:
            logger.debug(f"LangGraph store sync deferred: {store_err}")

        item = MemoryItemResponse(
            id=updated.id,
            userId=updated.userId,
            projectId=updated.projectId,
            type=updated.type.value if hasattr(updated.type, "value") else str(updated.type),
            key=updated.key,
            content=updated.content,
            importance=updated.importance,
            accessCount=updated.accessCount,
            lastAccessed=updated.lastAccessed.isoformat(),
            createdAt=updated.createdAt.isoformat(),
            updatedAt=updated.updatedAt.isoformat()
        )
        return {"success": True, "response": item}
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}


@router.delete("/{project_id}/memories/{memory_id}", response_model=ApiResponse[None])
async def delete_memory(
    project_id: str,
    memory_id: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """Deletes a memory record from both Prisma and LangGraph Store."""
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        existing = await prisma.memoryrecord.find_unique(where={"id": memory_id})
        if not existing or existing.userId != user_id:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"success": False, "error": "Memory record not found"}

        await prisma.memoryrecord.delete(where={"id": memory_id})

        # Remove from LangGraph Store across namespaces
        try:
            store = await get_memory_store()
            if store is not None:
                await store.adelete(("memories", user_id, project_id), key=existing.key)
                await store.adelete(("memories", user_id), key=existing.key)
        except Exception as store_err:
            logger.debug(f"LangGraph store delete deferred: {store_err}")

        return {"success": True, "response": None}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}
