from fastapi import status
from app.domain.entities.documents.documents_service import DocumentsService

class DocumentsController:
    @staticmethod
    async def upload_document(
        project_id: str,
        file,
        response,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                response.status_code = status.HTTP_401_UNAUTHORIZED
                return {"success": False, "error": "Unauthorized"}
            
            file_name = file.filename
            file_content = await file.read()
            content_type = file.content_type

            result = await DocumentsService.upload_document(
                project_id=project_id,
                user_id=user_id,
                file_name=file_name,
                file_content=file_content,
                content_type=content_type,
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
        except Exception as e:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_documents(
        project_id: str,
        response,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                response.status_code = status.HTTP_401_UNAUTHORIZED
                return {"success": False, "error": "Unauthorized"}
            
            result = await DocumentsService.get_documents(
                project_id=project_id,
                user_id=user_id,
            )
            if not result.get("success"):
                err = result.get("error")
                if err == "Project not found":
                    response.status_code = status.HTTP_404_NOT_FOUND
                elif err == "Unauthorized":
                    response.status_code = status.HTTP_403_FORBIDDEN
                else:
                    response.status_code = status.HTTP_400_BAD_REQUEST
            return result
        except Exception as e:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_document_by_id(
        project_id: str,
        document_id: str,
        response,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                response.status_code = status.HTTP_401_UNAUTHORIZED
                return {"success": False, "error": "Unauthorized"}
            
            result = await DocumentsService.get_document_by_id(
                project_id=project_id,
                document_id=document_id,
                user_id=user_id,
            )
            if not result.get("success"):
                err = result.get("error")
                if err in ["Project not found", "Document not found"]:
                    response.status_code = status.HTTP_404_NOT_FOUND
                elif err == "Unauthorized":
                    response.status_code = status.HTTP_403_FORBIDDEN
                else:
                    response.status_code = status.HTTP_400_BAD_REQUEST
            return result
        except Exception as e:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_document_download_url(
        project_id: str,
        document_id: str,
        response,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                response.status_code = status.HTTP_401_UNAUTHORIZED
                return {"success": False, "error": "Unauthorized"}
            
            result = await DocumentsService.get_document_download_url(
                project_id=project_id,
                document_id=document_id,
                user_id=user_id,
            )
            if not result.get("success"):
                err = result.get("error")
                if err in ["Project not found", "Document not found"]:
                    response.status_code = status.HTTP_404_NOT_FOUND
                elif err == "Unauthorized":
                    response.status_code = status.HTTP_403_FORBIDDEN
                else:
                    response.status_code = status.HTTP_400_BAD_REQUEST
            return result
        except Exception as e:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_document(
        project_id: str,
        document_id: str,
        response,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                response.status_code = status.HTTP_401_UNAUTHORIZED
                return {"success": False, "error": "Unauthorized"}
            
            result = await DocumentsService.delete_document(
                project_id=project_id,
                document_id=document_id,
                user_id=user_id,
            )
            if not result.get("success"):
                err = result.get("error")
                if err in ["Project not found", "Document not found"]:
                    response.status_code = status.HTTP_404_NOT_FOUND
                elif err == "Unauthorized":
                    response.status_code = status.HTTP_403_FORBIDDEN
                else:
                    response.status_code = status.HTTP_400_BAD_REQUEST
            return result
        except Exception as e:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": str(e)}
