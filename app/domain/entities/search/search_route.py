from fastapi import APIRouter, Depends, Response, status
from app.domain.entities.auth.auth_schema import ApiResponse
from app.domain.entities.search.search_schema import ChatRequest, ChatResponse
from app.domain.entities.search.search_controller import SearchController
from app.middleware.auth_middleware import auth_middleware

router = APIRouter(prefix="/projects", tags=["Search"])

@router.post("/{project_id}/chat", response_model=ApiResponse[ChatResponse])
async def chat_with_project(
    project_id: str,
    body: ChatRequest,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """
    Exposes RAG Chat Q&A endpoint.
    Retrieves project documents and relationship graph context to synthesize answers.
    """
    result = await SearchController.chat_with_project(project_id, body, credentials)
    if not result.get("success"):
        err = result.get("error")
        if err == "Project not found":
            response.status_code = status.HTTP_404_NOT_FOUND
        elif err == "Unauthorized":
            response.status_code = status.HTTP_403_FORBIDDEN
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
    return result
