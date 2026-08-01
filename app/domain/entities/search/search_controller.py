from app.core.database import prisma
from app.domain.entities.search.search_service import SearchService

class SearchController:
    @staticmethod
    async def chat_with_project(project_id: str, body, credentials: dict) -> dict:
        """
        Handles RAG chat for a project.
        Verifies project ownership before calling the search service.
        """
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}

            # Verify project existence and ownership
            project = await prisma.project.find_unique(where={"id": project_id})
            if not project:
                return {"success": False, "error": "Project not found"}
            
            if project.userId != user_id:
                return {"success": False, "error": "Unauthorized"}

            # Delegate to search service
            response = await SearchService.generate_rag_answer(
                project_id=project_id,
                query_text=body.prompt,
                document_id=body.documentId,
                limit=body.limit or 5
            )

            return {
                "success": True,
                "response": response
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
