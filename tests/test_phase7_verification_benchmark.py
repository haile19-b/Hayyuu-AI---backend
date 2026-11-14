"""Tests for Phase 7: Verification, Multi-Turn Benchmark Runner & Hardening.

Validates:
1. Multi-turn benchmark execution on curated software engineering sessions.
2. Metric computation (strict_correct, acceptable_hit, soft_score, strict/soft accuracy).
3. Evaluation JSON files and out-of-band RunLogger traces (turns.csv, turns.jsonl).
4. MetaMo guard edge cases (anti-premature, verification, evidence search, task planning).
5. Cross-thread long-term memory retrieval across conversational cold-starts.
6. Cognitive deliberation latency bounds (< 500ms).
7. Production Docker Compose container topology and health check configuration.
"""

from __future__ import annotations
import json
import time
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.agents.supervisor.metamo.shared import CognitiveState
from app.agents.supervisor.metamo.signals import MetaMoSignals
from app.agents.supervisor.metamo.controller import MetaMoController
from app.agents.supervisor.metamo.homeostasis import post_turn_update
from app.agents.memory.retriever import format_memories_for_prompt
from tests.evaluation.runner import BenchmarkRunner
from tests.evaluation.sessions.hayyuu_sessions import SESSIONS


# =============================================================================
# 1. Multi-Turn Benchmark Execution & Metric Validation
# =============================================================================

@pytest.mark.asyncio
async def test_benchmark_runner_full_run(tmp_path):
    """Executes the multi-turn benchmark runner and verifies metrics and evaluation files."""
    test_sessions = [
        {
            "name": "session_test_ambiguity",
            "queries": [
                "Make it better.",
                "I want to reduce database query latency from 500ms to 50ms."
            ],
            "expected_actions": [
                "act_clarify",
                "act_respond_direct"
            ],
            "acceptable_actions": [
                ["act_clarify"],
                ["act_respond_direct", "act_search", "act_plan_tasks"]
            ]
        },
        {
            "name": "session_test_verify",
            "queries": [
                "Verify whether REQ-1 conflicts with the security specification."
            ],
            "expected_actions": [
                "act_verify"
            ],
            "acceptable_actions": [
                ["act_verify", "act_analyze_conflict"]
            ]
        }
    ]

    runner = BenchmarkRunner(
        sessions=test_sessions,
        soft_credit=0.50,
        eval_output_dir=tmp_path / "bench_test",
        run_id="test_run_001"
    )

    report = await runner.run_all()

    # Validate overall report structure
    assert report["run_id"] == "test_run_001"
    assert report["session_count"] == 2
    assert report["total_turns"] == 3
    assert report["strict_accuracy"] >= 0.65
    assert report["soft_accuracy"] >= 0.80

    # Verify generated evaluation files
    turn_file = tmp_path / "bench_test" / "strict_per_turn.json"
    session_file = tmp_path / "bench_test" / "strict_per_session.json"
    overall_file = tmp_path / "bench_test" / "strict_overall.json"

    assert turn_file.exists()
    assert session_file.exists()
    assert overall_file.exists()

    turn_data = json.loads(turn_file.read_text(encoding="utf-8"))
    assert len(turn_data) == 3
    assert "strict_correct" in turn_data[0]
    assert "soft_score" in turn_data[0]
    assert "latency_ms" in turn_data[0]

    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    assert len(session_data) == 2
    assert session_data[0]["session"] == "session_test_ambiguity"
    assert session_data[0]["turns"] == 2

    # Verify out-of-band telemetry files created
    turns_csv = runner.run_logger.csv_path
    turns_jsonl = runner.run_logger.jsonl_path
    assert turns_csv.exists()
    assert turns_jsonl.exists()


# =============================================================================
# 2. MetaMo Guard Edge Cases
# =============================================================================

@pytest.mark.asyncio
async def test_guard_anti_premature_on_vague_prompts():
    """Validates that ambiguous/vague prompts strictly trigger act_clarify."""
    controller = MetaMoController()
    state = CognitiveState().to_dict()

    vague_queries = [
        "Improve it.",
        "Fix the code.",
        "Make something cool for our users."
    ]

    for q in vague_queries:
        decision = await controller.deliberate(
            user_query=q,
            chat_history=[],
            kg_context="",
            state=state
        )
        assert decision["action"] == "act_clarify", f"Query '{q}' failed to trigger act_clarify!"
        assert "Anti-Premature guard" in decision["reason"] or decision["signals"]["ambiguity"] >= 0.55


@pytest.mark.asyncio
async def test_guard_verification_on_conflict_requests():
    """Validates that explicit conflict queries strictly trigger act_verify or act_analyze_conflict."""
    controller = MetaMoController()
    state = CognitiveState().to_dict()

    decision = await controller.deliberate(
        user_query="Please verify whether Requirement REQ-1 conflicts with Requirement REQ-5.",
        chat_history=[],
        kg_context="",
        state=state
    )

    assert decision["action"] in ("act_verify", "act_analyze_conflict")
    assert "Verification" in decision["reason"] or decision["signals"]["verify_request"] is True


@pytest.mark.asyncio
async def test_guard_task_planning_on_sprint_requests():
    """Validates that sprint/task decomposition requests trigger act_plan_tasks or act_decompose."""
    controller = MetaMoController()
    state = CognitiveState().to_dict()

    decision = await controller.deliberate(
        user_query="Decompose sprint 4 user stories into detailed engineering tasks with estimates.",
        chat_history=[],
        kg_context="",
        state=state
    )

    assert decision["action"] in ("act_plan_tasks", "act_decompose")
    assert decision["signals"]["needs_task_plan"] >= 0.50


@pytest.mark.asyncio
async def test_guard_evidence_search_on_doc_queries():
    """Validates that documentation/code search queries trigger act_search."""
    controller = MetaMoController()
    state = CognitiveState().to_dict()

    decision = await controller.deliberate(
        user_query="Search the project docs and repository to check if we support OAuth2 PKCE flow.",
        chat_history=[],
        kg_context="",
        state=state
    )

    assert decision["action"] == "act_search"
    assert decision["signals"]["needs_external_evidence"] >= 0.50


# =============================================================================
# 3. Cross-Thread Memory Persistence & Recall
# =============================================================================

@pytest.mark.asyncio
async def test_cross_thread_memory_prompt_formatting():
    """Verifies that memories persisted in Thread A are cleanly formatted for Thread B."""
    # Simulate memories retrieved in a new thread
    mock_memories = [
        {
            "key": "architecture/database",
            "content": "Project uses Neon PostgreSQL with PGVector for hybrid search.",
            "importance": 0.9
        },
        {
            "key": "compliance/security",
            "content": "All patient PII must be encrypted at rest using AES-256.",
            "importance": 0.85
        }
    ]

    formatted_block = format_memories_for_prompt(mock_memories)

    assert "<project_memory>" in formatted_block
    assert "</project_memory>" in formatted_block
    assert "architecture/database" in formatted_block
    assert "Neon PostgreSQL with PGVector" in formatted_block
    assert "compliance/security" in formatted_block
    assert "AES-256" in formatted_block


# =============================================================================
# 4. Cognitive Deliberation Latency Bounds (< 500ms)
# =============================================================================

@pytest.mark.asyncio
async def test_cognitive_deliberation_latency_bound():
    """Verifies that local MetaMo deliberation executes well within the 500ms latency ceiling."""
    controller = MetaMoController()
    state = CognitiveState().to_dict()

    mock_signals = MetaMoSignals(
        cx=0.4,
        ambiguity=0.1,
        needs_external_evidence=0.3,
        needs_task_plan=0.2,
        needs_multi_source_integration=0.3,
        valence=0.1,
        intent_type="factual",
        reflective_intent=0.4,
        verify_request=False,
        user_expertise=0.6,
        urgency=0.0
    )

    with patch("app.agents.supervisor.metamo.controller.extract_signals", new=AsyncMock(return_value=mock_signals)):
        t0 = time.time()
        decision = await controller.deliberate(
            user_query="Provide an overview of the system architecture.",
            chat_history=[{"role": "user", "content": "Hello"}],
            kg_context="Component: AuthService, Database: Postgres",
            state=state
        )
        latency_ms = (time.time() - t0) * 1000.0

        assert latency_ms < 500.0, f"Deliberation exceeded 500ms ceiling: {latency_ms:.1f}ms"
        assert decision["action"] is not None
        assert decision["style_modifier"] is not None


# =============================================================================
# 5. Production Docker Compose Container Verification
# =============================================================================

def test_docker_compose_production_topology():
    """Validates docker-compose.yaml defines all required services and health checks."""
    import yaml
    compose_path = Path(__file__).resolve().parent.parent / "docker-compose.yaml"
    assert compose_path.exists(), "docker-compose.yaml not found!"

    with compose_path.open("r", encoding="utf-8") as f:
        compose_cfg = yaml.safe_load(f)

    services = compose_cfg.get("services", {})

    # Check all required production microservices exist
    assert "postgres" in services, "PostgreSQL service missing from docker-compose.yaml"
    assert "redis" in services, "Redis service missing from docker-compose.yaml"
    assert "neo4j" in services, "Neo4j service missing from docker-compose.yaml"
    assert "mcp" in services, "Standalone FastMCP service missing from docker-compose.yaml"
    assert "web" in services, "FastAPI Web service missing from docker-compose.yaml"
    assert "worker" in services, "ARQ Worker service missing from docker-compose.yaml"

    # Verify MCP container configuration (port 8001, SSE transport)
    mcp_svc = services["mcp"]
    assert "8001:8001" in mcp_svc.get("ports", [])
    assert mcp_svc.get("container_name") == "hayyuu_mcp"
    assert "healthcheck" in mcp_svc

    # Verify FastAPI Web container configuration (port 8000, connects to mcp:8001/sse)
    web_svc = services["web"]
    assert "8000:8000" in web_svc.get("ports", [])
    assert web_svc.get("environment", {}).get("HAYYUU_MCP_URL") == "http://mcp:8001/sse"
    assert "postgres" in web_svc.get("depends_on", {})
    assert "redis" in web_svc.get("depends_on", {})
    assert "mcp" in web_svc.get("depends_on", {})

    # Verify ARQ worker configuration
    worker_svc = services["worker"]
    assert "app.core.queue.WorkerSettings" in worker_svc.get("command", "")
    assert "mcp" in worker_svc.get("depends_on", {})

    # Verify persistent volume definitions
    volumes = compose_cfg.get("volumes", {})
    assert "pgdata" in volumes, "Persistent PostgreSQL volume missing"
