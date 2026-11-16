"""Node 4: Synthesize Node for Chat Agent.

Synthesizes multi-turn context, memory, and executed MCP tool evidence
into a coherent, factual response using the unified AI cascade engine.
"""

import logging
from typing import Dict, Any
from google.genai import types
from langchain_core.messages import AIMessage

from app.core.ai_cascade import generate_content_cascade
from app.agents.chat_agent.state import ChatAgentState

logger = logging.getLogger("uvicorn.error")


async def synthesize_node(state: ChatAgentState) -> Dict[str, Any]:
    """Node 4: Non-streaming synthesis fallback for graph execution."""
    system_instruction = state.get("system_instruction", "")
    gemini_contents = state.get("gemini_contents", [])

    try:
        response = await generate_content_cascade(
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
            timeout_seconds=30.0,
        )
        final_text = response.text.strip() if response and response.text else "Turn complete."
    except Exception as e:
        logger.error(f"[ChatAgent: synthesize] Cascade generation failed: {e}")
        final_text = "I apologize, but I encountered an error while synthesizing the response. Please try again."

    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
        "status": "synthesized"
    }
