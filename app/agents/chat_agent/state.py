"""Chat Agent State Definition.

Defines the LangGraph execution state schema for the Hayyuu Chat Agent.
"""

from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatAgentState(TypedDict):
    """LangGraph execution state schema for the Hayyuu Chat Agent."""
    # Conversation Turns Stream
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Session & Project Identifiers
    thread_id: str
    project_id: str
    user_id: str
    user_query: str
    
    # User Profile & Project Context
    user_profile: Optional[Dict[str, Any]]
    connected_repositories: List[Dict[str, Any]]
    project_documents: List[Dict[str, Any]]
    budgeted_context: Dict[str, Any]
    episodic_memories: List[Dict[str, Any]]
    
    # MetaMo Cognitive Deliberation
    cognitive_state: Dict[str, Any]
    deliberation_action: str
    deliberation_reason: str
    style_modifier: str
    score_top3: str
    signals: Dict[str, float]
    fast_lane: bool
    
    # Plan -> Do Investigation Loop
    investigation_plan: List[Dict[str, Any]]
    current_plan_step_idx: int
    is_plan_complete: bool
    structured_evidence: str

    # Dynamic MCP Tool Execution Loop
    pending_tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    tool_loop_count: int
    needs_tools: bool
    
    # LLM Prompt & Contents Buffer
    system_instruction: str
    gemini_contents: List[Any]
    
    # Final Output & Persistence
    final_response: str
    created_message_id: Optional[str]
    turn_completed: bool
    status: str
    error: Optional[str]
