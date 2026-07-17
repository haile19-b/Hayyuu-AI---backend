from typing import List
from fastapi import APIRouter, Depends, Response, UploadFile, File
from app.domain.entities.documents.documents_schema import (
    ApiResponse, DocumentResponse, DocumentDownloadResponse
)
from app.domain.entities.documents.documents_controller import DocumentsController
from app.middleware.auth_middleware import auth_middleware

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
