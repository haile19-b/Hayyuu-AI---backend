from app.domain.entities.documents.documents_service import DocumentsService

class DocumentsController:
    @staticmethod
    async def upload_document(
        project_id: str,
        file_name: str,
        file_content: bytes,
        content_type: str,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await DocumentsService.upload_document(
                project_id=project_id,
                user_id=user_id,
                file_name=file_name,
                file_content=file_content,
                content_type=content_type,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_documents(
        project_id: str,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await DocumentsService.get_documents(
                project_id=project_id,
                user_id=user_id,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_document_by_id(
        project_id: str,
        document_id: str,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await DocumentsService.get_document_by_id(
                project_id=project_id,
                document_id=document_id,
                user_id=user_id,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_document_download_url(
        project_id: str,
        document_id: str,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await DocumentsService.get_document_download_url(
                project_id=project_id,
                document_id=document_id,
                user_id=user_id,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_document(
        project_id: str,
        document_id: str,
        credentials: dict,
    ) -> dict:
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
            return await DocumentsService.delete_document(
                project_id=project_id,
                document_id=document_id,
                user_id=user_id,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}
