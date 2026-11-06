"""Tests for Phase 5: Multi-Agent Supervisor Graph and Orchestration.

Validates:
1. StateGraph node registration, conditional edges, and compilation.
2. Deliberation conditional routing edge across all 9 action routes.
3. Native Pydantic structured output schemas (RequirementsOutput, TaskPlanOutput).
4. Human-in-the-loop review routing condition (check_review_required).
5. Specialized action node execution (direct_response, clarify, requirements, task planning, verify, synthesize).
6. Deliberation node execution and out-of-band RunLogger telemetry generation.
7. Post-turn homeostasis node and thread state persistence.
"""

import sys
import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.supervisor.state import SupervisorState
from app.agents.supervisor.orchestrator import (
    build_supervisor_graph,
    get_compiled_supervisor,
    route_deliberation,
    check_review_required,
    RequirementsOutput,
    RequirementItem,
    TaskPlanOutput,
    TaskPlanItem,
    direct_response_node,
    clarify_node,
    requirements_agent_node,
    task_planner_node,
    verify_agent_node,
    synthesize_node,
    deliberate_node,
    post_turn_node,
)
from app.agents.supervisor.metamo.shared import CognitiveState
from app.agents.supervisor.metamo.signals import MetaMoSignals


# =============================================================================
# 1. StateGraph Architecture & Compilation Tests
# =============================================================================

def test_supervisor_graph_structure_and_nodes():
    """Verifies that all 12 nodes are registered and the graph compiles cleanly."""
    builder = build_supervisor_graph()

    expected_nodes = {
        "assemble_context",
        "deliberate",
        "act_respond_direct",
        "act_clarify",
        "act_search",
        "act_extract_requirements",
        "act_analyze_conflict",
        "act_build_knowledge",
        "act_plan_tasks",
        "act_verify",
        "act_synthesize",
        "human_review",
        "post_turn",
    }

    # Verify all expected nodes are in the builder
    for node_name in expected_nodes:
        assert node_name in builder.nodes, f"Node '{node_name}' missing from StateGraph"

    # Compile the graph without external checkpointer for unit verification
    compiled = builder.compile()
    assert compiled is not None


# =============================================================================
# 2. Deliberation Conditional Routing Edge Tests (9 Action Routes)
# =============================================================================

def test_route_deliberation_all_action_routes():
    """Verifies route_deliberation directs execution to all 9 candidate actions."""
    routes = [
        "act_respond_direct",
        "act_clarify",
        "act_search",
        "act_extract_requirements",
        "act_analyze_conflict",
        "act_build_knowledge",
        "act_plan_tasks",
        "act_verify",
        "act_synthesize",
    ]

    for action in routes:
        state = {"deliberation_action": action}
        assert route_deliberation(state) == action

    # Default fallback for unknown action
    assert route_deliberation({"deliberation_action": "act_unknown"}) == "act_respond_direct"


# =============================================================================
# 3. Native Structured Output Schemas Tests
# =============================================================================

def test_action_node_structured_output_schemas():
    """Verifies native Pydantic schemas for requirements and task planning nodes."""
    # 1. Requirements schema
    req = RequirementItem(
        temp_id="req_01",
        title="Async Event Processing",
        description="The backend must ingest SSE events asynchronously.",
        type="NON_FUNCTIONAL",
        priority="P0"
    )
    reqs_output = RequirementsOutput(
        summary="Extracted 1 non-functional requirement",
        requirements=[req]
    )
    req_json = reqs_output.model_dump_json()
    parsed_req = RequirementsOutput.model_validate_json(req_json)
    assert len(parsed_req.requirements) == 1
    assert parsed_req.requirements[0].priority == "P0"

    # 2. Task Plan schema
    task = TaskPlanItem(
        title="Implement Event Loop Policy",
        description="Configure WindowsSelectorEventLoopPolicy for async pg.",
        priority="HIGH",
        dependencies=["Setup DB Pool"]
    )
    plan_output = TaskPlanOutput(
        summary="Sprint plan for Windows async compatibility",
        tasks=[task]
    )
    plan_json = plan_output.model_dump_json()
    parsed_plan = TaskPlanOutput.model_validate_json(plan_json)
    assert len(parsed_plan.tasks) == 1
    assert parsed_plan.tasks[0].dependencies == ["Setup DB Pool"]


# =============================================================================
# 4. Human-In-The-Loop Conditional Routing Edge Tests
# =============================================================================

def test_check_review_required():
    """Verifies check_review_required routes to human_review only when review_required is True."""
    # When human review is requested
    assert check_review_required({"review_required": True}) == "human_review"

    # When routine turn without human review
    assert check_review_required({"review_required": False}) == "post_turn"
    assert check_review_required({}) == "post_turn"


# =============================================================================
# 5. Action Node Handlers Unit Tests
# =============================================================================

@pytest.mark.asyncio
async def test_direct_response_node():
    """Verifies direct_response_node execution."""
    fake_resp = MagicMock()
    fake_resp.text = "Here is the direct answer to your architecture query."

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=fake_resp)

    with patch("app.agents.supervisor.orchestrator.genAI", mock_client):
        state = {
            "user_query": "Explain the architecture",
            "style_modifier": "style_thorough",
            "budgeted_context": {"kg_context": "Sample project anchor"},
            "episodic_memories": []
        }
        res = await direct_response_node(state)
        assert "final_response" in res
        assert "architecture query" in res["final_response"]


@pytest.mark.asyncio
async def test_clarify_node():
    """Verifies clarify_node produces clarifying questions."""
    fake_resp = MagicMock()
    fake_resp.text = "1. Which database engine? 2. What is the target throughput?"

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=fake_resp)

    with patch("app.agents.supervisor.orchestrator.genAI", mock_client):
        state = {
            "user_query": "Build a database for it",
            "style_modifier": "style_cautious",
            "budgeted_context": {},
            "episodic_memories": []
        }
        res = await clarify_node(state)
        assert "final_response" in res
        assert "Which database engine" in res["final_response"]


@pytest.mark.asyncio
async def test_requirements_agent_node_structured_output():
    """Verifies requirements_agent_node formats structured requirement output."""
    fake_resp = MagicMock()
    fake_resp.text = json.dumps({
        "summary": "Core authentication requirements extracted",
        "requirements": [
            {
                "temp_id": "req_1",
                "title": "OAuth2 PKCE Login",
                "description": "System shall provide OAuth2 PKCE login flow.",
                "type": "FUNCTIONAL",
                "priority": "P0"
            }
        ]
    })

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=fake_resp)

    mock_prisma = MagicMock()
    mock_prisma.requirement.find_many = AsyncMock(return_value=[])

    with patch("app.agents.supervisor.orchestrator.genAI", mock_client):
        with patch("app.agents.supervisor.orchestrator.prisma", mock_prisma):
            state = {
                "user_query": "Specify user authentication with OAuth2",
                "project_id": "proj_123",
            }
            res = await requirements_agent_node(state)
            assert "OAuth2 PKCE Login" in res["final_response"]
            assert "subagent_output" in res


@pytest.mark.asyncio
async def test_task_planner_node_structured_output():
    """Verifies task_planner_node formats structured task plan output."""
    fake_resp = MagicMock()
    fake_resp.text = json.dumps({
        "summary": "Implementation tasks for user authentication",
        "tasks": [
            {
                "title": "Create Auth Route",
                "description": "Define FastAPI auth router endpoints.",
                "priority": "HIGH",
                "dependencies": []
            }
        ]
    })

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=fake_resp)

    with patch("app.agents.supervisor.orchestrator.genAI", mock_client):
        state = {
            "user_query": "Decompose authentication into developer tasks",
        }
        res = await task_planner_node(state)
        assert "Create Auth Route" in res["final_response"]
        assert "subagent_output" in res


# =============================================================================
# 6. Deliberate Node Execution & Out-Of-Band Telemetry Tests
# =============================================================================

@pytest.mark.asyncio
async def test_deliberate_node_execution():
    """Verifies deliberate_node invokes MetaMoController and produces telemetry."""
    mock_signals = MetaMoSignals(
        cx=0.3,
        ambiguity=0.1,
        needs_external_evidence=0.2,
        needs_task_plan=0.1,
    )

    with patch("app.agents.supervisor.orchestrator.RunLogger") as mock_logger_cls:
        mock_logger = MagicMock()
        mock_logger_cls.return_value = mock_logger

        state = {
            "thread_id": "thread_test_01",
            "user_query": "Hello, how does the system work?",
            "budgeted_context": {"tier1_chat_history": [], "kg_context": ""},
            "cognitive_state": CognitiveState().to_dict(),
        }

        res = await deliberate_node(state)
        assert "deliberation_action" in res
        assert "score_top3" in res
        assert "deliberation_reason" in res
        assert "style_modifier" in res

        # Verify RunLogger was invoked for out-of-band evaluation telemetry
        assert mock_logger.log_turn.called


# =============================================================================
# 7. Post-Turn Homeostasis Node Tests
# =============================================================================

@pytest.mark.asyncio
async def test_post_turn_node_homeostasis_and_persistence():
    """Verifies post_turn_node executes homeostasis and increments turn count."""
    mock_prisma = MagicMock()
    mock_prisma.conversationthread.update = AsyncMock(return_value=None)
    mock_created_msg = MagicMock()
    mock_created_msg.id = "msg_001"
    mock_prisma.chatmessage.create = AsyncMock(return_value=mock_created_msg)
    mock_prisma.execute_raw = AsyncMock(return_value=None)

    with patch("app.agents.supervisor.orchestrator.prisma", mock_prisma):
        cog_state = CognitiveState(turn_count=2).to_dict()
        state = {
            "thread_id": "thread_999",
            "cognitive_state": cog_state,
            "current_signals": {"cx": 0.5, "ambiguity": 0.2, "valence": 0.5},
            "deliberation_action": "act_respond_direct",
            "style_modifier": "style_balanced",
            "deliberation_reason": "Low-risk factual query",
            "final_response": "The architecture uses LangGraph supervisor.",
        }

        res = await post_turn_node(state)
        assert res["turn_count"] == 3
        assert len(res["messages"]) == 1
        assert mock_prisma.conversationthread.update.called
        assert mock_prisma.chatmessage.create.called
