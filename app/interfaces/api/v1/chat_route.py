import logging
from fastapi import APIRouter, Depends, Response, status

from app.core.database import prisma
from app.middleware.auth_middleware import auth_middleware
from app.domain.entities.auth.auth_schema import ApiResponse

from app.agents.search_agent.schemas import ChatRequest, ChatResponse
from app.agents.search_agent.state import SearchAgentState
from app.agents.search_agent.graph import search_agent_graph

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/projects", tags=["Chat"])


@router.post("/{project_id}/chat", response_model=ApiResponse[ChatResponse])
async def chat_with_project(
    project_id: str,
    body: ChatRequest,
    response: Response,
    credentials = Depends(auth_middleware)
):
    """
    Exposes RAG Chat Q&A endpoint.
    Orchestrates the workflow via the modular LangGraph Search Agent.
    """
    try:
        user_id = credentials.get("id") if credentials else None
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"success": False, "error": "Unauthorized"}

        # Verify project existence and user ownership
        project = await prisma.project.find_unique(where={"id": project_id})
        if not project:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"success": False, "error": "Project not found"}
        
        if project.userId != user_id:
            response.status_code = status.HTTP_403_FORBIDDEN
            return {"success": False, "error": "Unauthorized"}

        # Invoke the LangGraph Search Agent
        logger.info(f"Invoking search agent graph for project {project_id}")
        state = SearchAgentState(
            project_id=project_id,
            query_text=body.prompt,
            document_id=body.documentId,
            limit=body.limit or 5
        )

        result_state = await search_agent_graph.ainvoke(state)

        # Check for errors in state
        if result_state.get("errors"):
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "success": False,
                "error": "; ".join(result_state.get("errors"))
            }

        # Compile final response structure
        response_data = ChatResponse(
            answer=result_state.get("answer", ""),
            sources=result_state.get("chunks", []),
            nodes=result_state.get("graph_nodes", [])
        )

        return {
            "success": True,
            "response": response_data
        }

    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"success": False, "error": str(e)}
