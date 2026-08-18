import uuid
from app.core.database import prisma
from app.core.storage import storage_utility
from app.domain.entities.projects.projects_service import ProjectsService
from prisma.enums import DocumentStatus
from app.core.queue import enqueue_document_analysis

class DocumentsService:
    @staticmethod
    async def upload_document(
        project_id: str,
        user_id: str,
        file_name: str,
        file_content: bytes,
        content_type: str,
    ) -> dict:
        try:
            # 1. Verify project ownership
            project_res = await ProjectsService.get_project_by_id(project_id, user_id)
            if not project_res.get("success"):
                return project_res

            # 2. Upload file to R2 / S3 with local disk fallback
            unique_id = uuid.uuid4().hex
            file_path = f"projects/{project_id}/documents/{unique_id}_{file_name}"
            
            try:
                await storage_utility.upload_file(file_content, file_path, content_type)
            except Exception as s3_err:
                import os
                print(f"S3/R2 upload failed: {s3_err}. Falling back to local storage.")
                local_dir = os.path.join("uploads", project_id)
                os.makedirs(local_dir, exist_ok=True)
                local_path = os.path.join(local_dir, f"{unique_id}_{file_name}")
                with open(local_path, "wb") as f:
                    f.write(file_content)
                file_path = local_path

            # 3. Save document record in the database
            document = await prisma.document.create(
                data={
                    "projectId": project_id,
                    "name": file_name,
                    "filePath": file_path,
                    "fileType": content_type,
                    "sizeBytes": len(file_content),
                    "status": DocumentStatus.PENDING,
                }
            )

            # 4. Trigger background analysis job
            await enqueue_document_analysis(document.id, project_id)

            return {
                "success": True,
                "response": document
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_documents(
        project_id: str,
        user_id: str,
    ) -> dict:
        try:
            # Verify project ownership
            project_res = await ProjectsService.get_project_by_id(project_id, user_id)
            if not project_res.get("success"):
                return project_res

            documents = await prisma.document.find_many(
                where={"projectId": project_id}
            )
            return {
                "success": True,
                "response": documents
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_document_by_id(
        project_id: str,
        document_id: str,
        user_id: str,
    ) -> dict:
        try:
            # Verify project ownership
            project_res = await ProjectsService.get_project_by_id(project_id, user_id)
            if not project_res.get("success"):
                return project_res

            document = await prisma.document.find_unique(
                where={"id": document_id}
            )
            if not document or document.projectId != project_id:
                return {"success": False, "error": "Document not found"}

            return {
                "success": True,
                "response": document
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_document_download_url(
        project_id: str,
        document_id: str,
        user_id: str,
    ) -> dict:
        try:
            # Verify project ownership
            project_res = await ProjectsService.get_project_by_id(project_id, user_id)
            if not project_res.get("success"):
                return project_res

            document = await prisma.document.find_unique(
                where={"id": document_id}
            )
            if not document or document.projectId != project_id:
                return {"success": False, "error": "Document not found"}

            # Generate signed URL
            download_url = await storage_utility.generate_download_url(document.filePath)

            return {
                "success": True,
                "response": {
                    "downloadUrl": download_url
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_document(
        project_id: str,
        document_id: str,
        user_id: str,
    ) -> dict:
        try:
            # Verify project ownership
            project_res = await ProjectsService.get_project_by_id(project_id, user_id)
            if not project_res.get("success"):
                return project_res

            document = await prisma.document.find_unique(
                where={"id": document_id}
            )
            if not document or document.projectId != project_id:
                return {"success": False, "error": "Document not found"}

            # Delete file from R2 / S3
            try:
                await storage_utility.delete_file(document.filePath)
            except Exception as s3_err:
                # Log or handle S3 deletion failure, but keep going with db delete
                # or optionally bubble up. Let's make it robust.
                pass

            # Delete document record from database
            await prisma.document.delete(
                where={"id": document_id}
            )

            return {
                "success": True,
                "response": None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
