from typing import List
from fastapi import APIRouter, Depends, Response, UploadFile, File, WebSocket, WebSocketDisconnect
import json
import asyncio
import logging
from app.domain.entities.documents.documents_schema import (
    ApiResponse, DocumentResponse, DocumentDownloadResponse
)
from app.domain.entities.documents.documents_controller import DocumentsController
from app.middleware.auth_middleware import auth_middleware
from app.core import queue
from app.core.database import prisma
from prisma.enums import DocumentStatus

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/projects", tags=["Documents"])

@router.post("/{projectId}/documents", response_model=ApiResponse[DocumentResponse])
async def upload_document(
    projectId: str,
    response: Response,
    file: UploadFile = File(...),
    credentials = Depends(auth_middleware)
):
    return await DocumentsController.upload_document(
        project_id=projectId,
        file=file,
        response=response,
        credentials=credentials
    )

@router.get("/{projectId}/documents", response_model=ApiResponse[List[DocumentResponse]])
async def get_documents(
    projectId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    return await DocumentsController.get_documents(
        project_id=projectId,
        response=response,
        credentials=credentials
    )

@router.get("/{projectId}/documents/{documentId}", response_model=ApiResponse[DocumentResponse])
async def get_document_by_id(
    projectId: str,
    documentId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    return await DocumentsController.get_document_by_id(
        project_id=projectId,
        document_id=documentId,
        response=response,
        credentials=credentials
    )

@router.get("/{projectId}/documents/{documentId}/download", response_model=ApiResponse[DocumentDownloadResponse])
async def get_document_download_url(
    projectId: str,
    documentId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    return await DocumentsController.get_document_download_url(
        project_id=projectId,
        document_id=documentId,
        response=response,
        credentials=credentials
    )

@router.delete("/{projectId}/documents/{documentId}", response_model=ApiResponse[None])
async def delete_document(
    projectId: str,
    documentId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    return await DocumentsController.delete_document(
        project_id=projectId,
        document_id=documentId,
        response=response,
        credentials=credentials
    )

@router.websocket("/{projectId}/documents/{documentId}/stream")
async def stream_document_progress(
    websocket: WebSocket,
    projectId: str,
    documentId: str
):
    """WebSocket endpoint to subscribe to Redis Pub/Sub progress channel for a document and stream updates."""
    await websocket.accept()
    logger.info(f"WebSocket client connected to stream for document {documentId}")
    
    if queue.redis_pool is None:
        await queue.connect_redis()
        
    channel_name = f"document_analysis:{documentId}"
    pubsub = queue.redis_pool.pubsub()
    await pubsub.subscribe(channel_name)
    
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                payload = json.loads(message["data"])
                await websocket.send_json(payload)
                
                # Stop streaming if completion or failure state is published
                if payload.get("status") in ["COMPLETED", "FAILED"]:
                    logger.info(f"Analysis completed/failed for {documentId}, closing stream.")
                    break
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for document {documentId}")
    except Exception as e:
        logger.error(f"Error in WebSocket streaming for {documentId}: {e}")
        try:
            await websocket.send_json({"status": "FAILED", "message": f"Streaming error: {str(e)}"})
        except Exception:
            pass
    finally:
        await pubsub.unsubscribe(channel_name)
        await pubsub.close()

@router.post("/{projectId}/documents/{documentId}/retry", response_model=ApiResponse[None])
async def retry_document_analysis(
    projectId: str,
    documentId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """Resume a failed document analysis pipeline from the last completed node checkpoint."""
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = 401
            return {"success": False, "error": "Unauthorized"}
            
        # 1. Fetch document and verify project mapping
        document = await prisma.document.find_unique(where={"id": documentId})
        if not document or document.projectId != projectId:
            response.status_code = 404
            return {"success": False, "error": "Document not found"}
            
        # 2. Reset status to PENDING
        await prisma.document.update(
            where={"id": documentId},
            data={"status": DocumentStatus.PENDING}
        )
        
        # 3. Enqueue the analysis job.
        # The LangGraph execution function (run_workflow) will automatically look up
        # the thread ID (document ID) checkpoint and resume from where it stopped.
        await queue.enqueue_document_analysis(documentId, projectId)
        
        return {"success": True, "response": None}
    except Exception as e:
        logger.error(f"Error triggering retry for document {documentId}: {e}")
        response.status_code = 400
        return {"success": False, "error": str(e)}


