"""Node 1: Assemble Context Node for Chat Agent.

Gathers authenticated user profile, project details, connected repositories,
uploaded documents, token-budgeted conversational turns, and long-term episodic memories.
"""

import json
import logging
from typing import Dict, Any

from app.core.database import prisma, ensure_db_connected
from app.agents.memory.retriever import retrieve_memories
from app.agents.supervisor.context_budget import assemble_context_with_budget
from app.agents.supervisor.metamo.shared import CognitiveState
from app.agents.chat_agent.state import ChatAgentState

logger = logging.getLogger("uvicorn.error")


async def assemble_context_node(state: ChatAgentState) -> Dict[str, Any]:
    """Node 1: Assembles multi-tier token-budgeted context, user profile, and memories."""
    thread_id = state["thread_id"]
    project_id = state["project_id"]
    user_id = state["user_id"]
    user_query = state["user_query"]

    await ensure_db_connected()

    # 1. Retrieve episodic memories
    memories = []
    try:
        memories = await retrieve_memories(
            user_id=user_id,
            project_id=project_id,
            query=user_query,
            limit=10
        )
    except Exception as mem_err:
        logger.warning(f"[ChatAgent: assemble_context] Memory retrieval deferred: {mem_err}")

    # 2. Token-budgeted context assembly (Tier 1 recent history, Tier 2 semantic search, KG context)
    budgeted = await assemble_context_with_budget(
        thread_id=thread_id,
        project_id=project_id,
        user_query=user_query,
        tier1_tokens=2000,
        tier2_tokens=1500,
        kg_tokens=1500,
        semantic_search=True,
        memories=memories,
    )

    # 3. Fetch authenticated user profile for persistent context awareness
    user_profile = state.get("user_profile")
    if not user_profile and user_id:
        try:
            user_rec = await prisma.user.find_unique(where={"id": user_id})
            if user_rec:
                user_profile = {
                    "fullName": user_rec.fullName,
                    "userName": user_rec.userName,
                    "email": user_rec.email,
                }
        except Exception as u_err:
            logger.debug(f"[ChatAgent: assemble_context] User profile lookup deferred: {u_err}")

    # 4. Fetch Project with connected repositories and uploaded documents
    connected_repos = []
    project_docs = []
    try:
        project_rec = await prisma.project.find_unique(
            where={"id": project_id},
            include={"repositories": True, "documents": True}
        )
        if project_rec:
            if project_rec.repositories:
                connected_repos = [
                    {
                        "id": r.id,
                        "name": r.name,
                        "owner": r.owner,
                        "fullName": r.fullName,
                        "htmlUrl": r.htmlUrl,
                        "defaultBranch": r.defaultBranch,
                        "isActive": r.isActive
                    }
                    for r in project_rec.repositories
                ]
            if project_rec.documents:
                project_docs = [
                    {
                        "id": d.id,
                        "name": d.name,
                        "fileType": d.fileType,
                        "status": str(d.status),
                        "sizeBytes": d.sizeBytes
                    }
                    for d in project_rec.documents
                ]
    except Exception as p_err:
        logger.debug(f"[ChatAgent: assemble_context] Project entities lookup deferred: {p_err}")

    # 5. Load thread cognitive state
    cog_state_dict = state.get("cognitive_state")
    if not cog_state_dict:
        try:
            thread_rec = await prisma.conversationthread.find_unique(where={"id": thread_id})
            if thread_rec and thread_rec.cognitiveState:
                raw_cog = thread_rec.cognitiveState
                cog_state_dict = json.loads(raw_cog) if isinstance(raw_cog, str) else raw_cog
        except Exception as cog_err:
            logger.debug(f"[ChatAgent: assemble_context] Thread cognitiveState load deferred: {cog_err}")

    if not cog_state_dict:
        cog_state_dict = CognitiveState().to_dict()

    return {
        "budgeted_context": budgeted,
        "episodic_memories": memories,
        "user_profile": user_profile,
        "connected_repositories": connected_repos,
        "project_documents": project_docs,
        "cognitive_state": cog_state_dict,
        "status": "context_assembled"
    }
