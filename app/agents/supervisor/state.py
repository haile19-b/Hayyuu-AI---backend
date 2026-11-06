"""Supervisor Agent State Definition.

Defines the LangGraph execution state for the MetaMo Cognitive Supervisor.
"""

from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    """Execution state schema for the cognitive supervisor graph."""
    # Conversation message stream
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Session Context
    thread_id: str
    project_id: str
    user_id: str
    user_query: str

    # MetaMo Cognitive Deliberation State
    cognitive_state: Dict[str, Any]
    current_signals: Dict[str, float]
    deliberation_action: str
    score_top3: str
    deliberation_reason: str
    style_modifier: str
    fast_lane: bool

    # Retrieved Multi-Tier Context
    budgeted_context: Dict[str, Any]
    episodic_memories: List[Dict[str, Any]]

    # Subagent Execution & Tool Output
    subagent_output: Optional[str]
    tool_calls_executed: List[Dict[str, Any]]

    # Generated Output & Metadata
    final_response: str
    turn_count: int

    # Human-In-The-Loop Review
    review_required: Optional[bool]
    review_status: Optional[Dict[str, Any]]
