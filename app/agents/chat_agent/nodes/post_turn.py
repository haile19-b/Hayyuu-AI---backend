"""Node 5: Post-Turn Node for Chat Agent.

Handles post-turn cognitive homeostasis updates, database persistence
for assistant chat messages, embedding calculation, and background memory consolidation.
"""

import json
import logging
from typing import Dict, Any

from app.core.database import prisma, ensure_db_connected
from prisma.enums import MessageRole
from app.agents.supervisor.metamo.homeostasis import post_turn_update
from app.agents.chat_agent.state import ChatAgentState

logger = logging.getLogger("uvicorn.error")


async def post_turn_node(state: ChatAgentState) -> Dict[str, Any]:
    """Node 5: Post-turn homeostasis, message persistence, embeddings, and memory consolidation."""
    thread_id = state["thread_id"]
    project_id = state.get("project_id")
    user_id = state["user_id"]
    final_response = state.get("final_response", "").strip()
    action = state.get("deliberation_action", "act_respond_direct")
    style_mod = state.get("style_modifier", "style_balanced")
    signals = state.get("signals", {})
    cog_state = state.get("cognitive_state", {})

    # 1. Update MetaMo homeostasis & turn counters
    updated_cog_state = post_turn_update(
        state=cog_state,
        signals=signals,
        decision={"action": action, "style_modifier": style_mod},
        turn_success=bool(final_response)
    )

    created_msg_id = None
    try:
        await ensure_db_connected()

        # 2. Persist updated thread cognitive state
        await prisma.conversationthread.update(
            where={"id": thread_id},
            data={"cognitiveState": json.dumps(updated_cog_state)}
        )

        # 3. Persist assistant ChatMessage (zero telemetry bloat)
        created_msg = await prisma.chatmessage.create(
            data={
                "threadId": thread_id,
                "role": MessageRole.ASSISTANT,
                "content": final_response,
                "cognitiveAction": action,
                "styleModifier": style_mod,
            }
        )
        created_msg_id = created_msg.id

        # 4. Generate vector embedding asynchronously
        try:
            from app.infrastructure.ai.gemini_embedder import GeminiEmbedder
            embedder = GeminiEmbedder()
            vec = embedder.embed_text(final_response, user_query=False)
            vec_lit = "[" + ",".join(str(x) for x in vec) + "]"
            await prisma.execute_raw(
                f"UPDATE chat_messages SET embedding = '{vec_lit}'::vector WHERE id = $1;",
                created_msg.id
            )
        except Exception as embed_err:
            logger.debug(f"[ChatAgent: post_turn] Embedding generation deferred: {embed_err}")

    except Exception as persist_err:
        logger.error(f"[ChatAgent: post_turn] Error persisting assistant message: {persist_err}")

    # 5. Trigger autonomous background episodic memory consolidation
    try:
        from app.core.queue import enqueue_memory_consolidation
        await enqueue_memory_consolidation(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id
        )
    except Exception as q_err:
        logger.debug(f"[ChatAgent: post_turn] Memory consolidation trigger deferred: {q_err}")

    return {
        "cognitive_state": updated_cog_state,
        "created_message_id": created_msg_id,
        "turn_completed": True,
        "status": "completed"
    }
