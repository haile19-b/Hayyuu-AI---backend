"""Chat Agent Data Schemas.

Defines input/output schemas, tool calling structures, and streaming event payloads
for the Hayyuu Chat Agent.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatTurnInput(BaseModel):
    """Input payload for initiating a chat agent turn."""
    project_id: str = Field(..., description="Active project UUID")
    user_id: str = Field(..., description="Authenticated user UUID")
    thread_id: str = Field(..., description="Active conversation thread UUID")
    prompt: str = Field(..., description="User's query or message content")


class ToolExecutionRecord(BaseModel):
    """Record of a tool invocation executed during an agent turn."""
    tool_name: str
    arguments: Dict[str, Any]
    result: str
    execution_time_ms: float = 0.0
    status: str = "success"  # "success" | "failed"


class PlanStep(BaseModel):
    """Discrete step or sub-goal within an investigation plan."""
    id: str = Field(..., description="Unique step identifier, e.g. 'step_1'")
    description: str = Field(..., description="Actionable goal of this step")
    tool_hint: Optional[str] = Field(None, description="Suggested MCP tool name to execute")
    status: str = Field(default="pending", description="'pending', 'in_progress', 'completed', or 'failed'")
    result_summary: Optional[str] = Field(None, description="Key fact or summary retrieved from this step")


class InvestigationPlan(BaseModel):
    """Autonomous plan breaking down information needs before final response generation."""
    reasoning: str = Field(..., description="Rationale for why these steps are needed to answer the query")
    steps: List[PlanStep] = Field(default_factory=list, description="Ordered list of investigation steps")
    needs_tools: bool = Field(default=True, description="Whether tools are required or query can be answered directly")


class ChatTurnOutput(BaseModel):
    """Output summary of a completed chat agent turn."""
    thread_id: str
    message_id: Optional[str] = None
    response_text: str
    deliberation_action: str
    style_modifier: str
    turn_count: int
    tool_calls: List[ToolExecutionRecord] = Field(default_factory=list)
    status: str = "completed"
