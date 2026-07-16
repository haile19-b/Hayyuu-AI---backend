from typing import List
from fastapi import APIRouter, Depends, Response, status, UploadFile, File
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
    file_content = await file.read()
    result = await DocumentsController.upload_document(
        project_id=projectId,
        file_name=file.filename,
        file_content=file_content,
        content_type=file.content_type,
        credentials=credentials
    )
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    else:
        response.status_code = status.HTTP_201_CREATED
    return result

@router.get("/{projectId}/documents", response_model=ApiResponse[List[DocumentResponse]])
async def get_documents(
    projectId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await DocumentsController.get_documents(projectId, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.get("/{projectId}/documents/{documentId}", response_model=ApiResponse[DocumentResponse])
async def get_document_by_id(
    projectId: str,
    documentId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await DocumentsController.get_document_by_id(projectId, documentId, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err in ["Project not found", "Document not found"]:
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.get("/{projectId}/documents/{documentId}/download", response_model=ApiResponse[DocumentDownloadResponse])
async def get_document_download_url(
    projectId: str,
    documentId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await DocumentsController.get_document_download_url(projectId, documentId, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err in ["Project not found", "Document not found"]:
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result

@router.delete("/{projectId}/documents/{documentId}", response_model=ApiResponse[None])
async def delete_document(
    projectId: str,
    documentId: str,
    response: Response,
    credentials = Depends(auth_middleware)
):
    result = await DocumentsController.delete_document(projectId, documentId, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err in ["Project not found", "Document not found"]:
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result
