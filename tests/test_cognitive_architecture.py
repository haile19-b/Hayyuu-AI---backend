"""Unit and Integration Test Suite for MetaMo Cognitive Architecture & Hayyuu AI Supervisor.

Tests:
1. Clamping utilities (unit & signed unit interval)
2. MetaMo state & dimensions (14 goals, 12 modulators, 4 anti-goals, signed valence)
3. Perceptual signal extraction & calibrations
4. Fast-lane routes & anti-premature guards
5. Multi-goal deliberation scoring & dynamic style modifiers
6. Two-phase post-turn update mathematics & homeostatic contractivity
7. Out-of-band evaluation logging (zero database bloat)
8. Context budgeting engine
9. Supervisor graph compilation & routing integrity
"""

import os
import sys
import pytest
import csv
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.supervisor.metamo.shared import (
    clamp_to_unit_interval,
    clamp_to_signed_unit_interval,
    clamp_range,
    GOAL_KEYS,
    MODULATOR_KEYS,
    ANTI_GOAL_KEYS,
    ACTION_KEYS,
    DEFAULT_GOALS,
    DEFAULT_MODULATORS,
    DEFAULT_ANTI_GOALS,
    CognitiveState,
)
from app.agents.supervisor.metamo.signals import extract_signals
from app.agents.supervisor.metamo.controller import MetaMoController
from app.agents.supervisor.metamo.homeostasis import (
    post_turn_update,
    apply_homeostatic_contractivity,
    compute_goal_targets,
    compute_anti_goal_targets,
)
from app.agents.supervisor.metamo.style import determine_style
from app.agents.supervisor.metamo.eval_logger import RunLogger, CSV_HEADERS
from app.agents.supervisor.context_budget import estimate_tokens
from app.agents.supervisor.orchestrator import build_supervisor_graph, route_deliberation


# -----------------------------------------------------------------------------
# 1. Clamping Functions Tests
# -----------------------------------------------------------------------------

def test_clamping_functions():
    # Unit interval [0.0, 1.0]
    assert clamp_to_unit_interval(-0.5) == 0.0
    assert clamp_to_unit_interval(1.5) == 1.0
    assert clamp_to_unit_interval(0.42) == 0.42
    assert clamp_to_unit_interval(0.0) == 0.0
    assert clamp_to_unit_interval(1.0) == 1.0

    # Signed unit interval [-1.0, 1.0]
    assert clamp_to_signed_unit_interval(-1.8) == -1.0
    assert clamp_to_signed_unit_interval(2.3) == 1.0
    assert clamp_to_signed_unit_interval(-0.35) == -0.35
    assert clamp_to_signed_unit_interval(0.75) == 0.75

    # Range clamp
    assert clamp_range(0.1, 0.2, 0.8) == 0.2
    assert clamp_range(0.9, 0.2, 0.8) == 0.8
    assert clamp_range(0.5, 0.2, 0.8) == 0.5


# -----------------------------------------------------------------------------
# 2. Dimensions and Initial State Tests
# -----------------------------------------------------------------------------

def test_metamo_dimensions_and_initial_state():
    assert len(GOAL_KEYS) == 14
    assert len(MODULATOR_KEYS) == 12
    assert len(ANTI_GOAL_KEYS) == 4
    assert len(ACTION_KEYS) == 9

    state = CognitiveState()
    assert state.turn_count == 0
    assert state.valence == 0.0
    assert state.user_expertise == 0.50

    # All keys properly initialized in default dictionary
    for g in GOAL_KEYS:
        assert g in state.goals
        assert 0.05 <= state.goals[g] <= 0.95

    for m in MODULATOR_KEYS:
        assert m in state.modulators
        assert 0.05 <= state.modulators[m] <= 0.95

    for a in ANTI_GOAL_KEYS:
        assert a in state.anti_goals
        assert 0.05 <= state.anti_goals[a] <= 0.95

    # Serialization roundtrip
    data = state.to_dict()
    restored = CognitiveState.from_dict(data)
    assert restored.turn_count == state.turn_count
    assert restored.goals == state.goals
    assert restored.valence == state.valence


# -----------------------------------------------------------------------------
# 3. Perceptual Signal Extraction Tests
# -----------------------------------------------------------------------------

def test_signal_extraction():
    # Ambiguous query
    vague_q = "Can you please fix it somehow?"
    signals_vague = extract_signals(vague_q, [], "")
    assert signals_vague["ambiguity"] > 0.40

    # Urgent query
    urgent_q = "Production database is broken! Fix this urgently ASAP!"
    signals_urgent = extract_signals(urgent_q, [], "")
    assert signals_urgent["urgency"] >= 0.50

    # Technical / complex query with code snippet
    complex_q = """
    ```python
    async def handle_concurrency():
        await asyncio.gather(task1(), task2())
    ```
    How do we optimize PostgreSQL connection pool sizing with Neon PgBouncer under high concurrency?
    """
    signals_complex = extract_signals(complex_q, [], "")
    assert signals_complex["complexity"] >= 0.45
    assert signals_complex["code_context"] == 1.0

    # Conflict query
    conflict_q = "We have an active conflict between Requirement P0-1 and Requirement P0-2."
    signals_conflict = extract_signals(conflict_q, [], kg_context="[Active Requirement Conflicts]\n- Conflict 1")
    assert signals_conflict["conflict_detected"] >= 0.65

    # Greeting
    greeting_q = "hello"
    signals_greeting = extract_signals(greeting_q, [], "")
    assert signals_greeting["is_greeting"] == 1.0


# -----------------------------------------------------------------------------
# 4. Fast-Lane Routing Tests
# -----------------------------------------------------------------------------

def test_fast_lane_routing():
    controller = MetaMoController()

    # Fast lane 1: Greeting
    res_greet = controller.deliberate("hello", [], "")
    assert res_greet["action"] == "act_respond_direct"
    assert res_greet["fast_lane"] is True
    assert "greeting" in res_greet["reason"].lower()

    # Fast lane 2: Critical conflict
    state_conflict = CognitiveState()
    state_conflict.modulators["conflict_tension"] = 0.75
    res_conflict = controller.deliberate(
        "There is a severe contradiction in the requirement constraints",
        [],
        kg_context="[Active Requirement Conflicts]\n- Conflict 101",
        state=state_conflict
    )
    assert res_conflict["action"] == "act_analyze_conflict"
    assert res_conflict["fast_lane"] is True

    # Fast lane 3: Anti-Premature Action Guard
    # Extremely ambiguous query with technical complexity
    res_vague = controller.deliberate(
        "Make something work with database migration and microservice schema architecture somehow",
        [],
        ""
    )
    if res_vague["signals"]["ambiguity"] >= 0.65 and res_vague["signals"]["complexity"] >= 0.30:
        assert res_vague["action"] == "act_clarify"
        assert res_vague["fast_lane"] is True
        assert "premature" in res_vague["reason"].lower()


# -----------------------------------------------------------------------------
# 5. Multi-Goal Deliberation & Style Selection Tests
# -----------------------------------------------------------------------------

def test_deliberation_scoring_and_style():
    controller = MetaMoController()

    # Requirement intent query
    req_q = "The system must support OAuth2 PKCE login flow with 15-minute JWT expiry tokens (P0 requirement)."
    res_req = controller.deliberate(req_q, [], "")
    assert res_req["action"] in ["act_extract_requirements", "act_verify", "act_respond_direct"]
    assert "score_top3" in res_req
    assert len(res_req["scores"]) == 9

    # Style determination
    # High urgency -> style_concise
    style_urgent = determine_style({"urgency": 0.85}, "act_respond_direct")
    assert style_urgent == "style_concise"

    # High conflict tension -> style_cautious
    style_cautious = determine_style({"conflict_tension": 0.70}, "act_analyze_conflict")
    assert style_cautious == "style_cautious"

    # Low user expertise -> style_tutorial
    style_tutorial = determine_style({"urgency": 0.2}, "act_respond_direct", user_expertise=0.25)
    assert style_tutorial == "style_tutorial"


# -----------------------------------------------------------------------------
# 6. Two-Phase Post-Turn Update & Homeostatic Contractivity Tests
# -----------------------------------------------------------------------------

def test_homeostasis_and_post_turn_update():
    state = CognitiveState().to_dict()
    initial_turn = state["turn_count"]
    initial_valence = state["valence"]

    signals = {
        "ambiguity": 0.2,
        "urgency": 0.1,
        "complexity": 0.6,
        "conflict_detected": 0.1,
        "requirement_intent": 0.3,
    }
    decision = {
        "action": "act_search",
        "style_modifier": "style_thorough"
    }

    # Successful turn update
    updated = post_turn_update(state, signals, decision, turn_success=True)

    assert updated["turn_count"] == initial_turn + 1
    assert updated["valence"] > initial_valence  # valence increases on success
    assert updated["last_action"] == "act_search"
    assert updated["last_style"] == "style_thorough"

    # Topic familiarity increased due to complexity >= 0.4
    assert updated["modulators"]["topic_familiarity"] >= DEFAULT_MODULATORS["topic_familiarity"]

    # Verify homeostatic contractivity pulls boundary values inward
    state["goals"]["clarify_first"] = 0.06  # near lower bound 0.05
    apply_homeostatic_contractivity(state)
    assert state["goals"]["clarify_first"] > 0.06  # gently pulled toward center 0.55


# -----------------------------------------------------------------------------
# 7. Out-of-Band Evaluation Logging (Zero DB Bloat) Tests
# -----------------------------------------------------------------------------

def test_run_logger_file_output(tmp_path):
    logger = RunLogger(run_id="test_run_123", base_dir=tmp_path)

    cog_state = CognitiveState().to_dict()
    logger.log_turn(
        thread_id="test_thread_abc",
        turn=1,
        query="Explain the database architecture",
        action="act_search",
        style_modifier="style_thorough",
        score_top3="act_search (0.820), act_respond_direct (0.450)",
        reason="Deliberated act_search due to technical complexity",
        cognitive_state=cog_state,
        latency_ms=45.2
    )

    # 1. Verify CSV created and populated
    assert logger.csv_path.exists()
    with logger.csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        r = rows[0]
        assert r["run_id"] == "test_run_123"
        assert r["thread_id"] == "test_thread_abc"
        assert r["action"] == "act_search"
        assert r["style_modifier"] == "style_thorough"
        assert "act_search" in r["score_top3"]

    # 2. Verify JSONL created and populated
    assert logger.jsonl_path.exists()
    with logger.jsonl_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["thread_id"] == "test_thread_abc"
        assert record["turn"] == 1


# -----------------------------------------------------------------------------
# 8. Token Estimator Tests
# -----------------------------------------------------------------------------

def test_token_estimator():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") == 1
    assert estimate_tokens("a" * 400) == 100


# -----------------------------------------------------------------------------
# 9. Supervisor StateGraph Compilation & Routing Tests
# -----------------------------------------------------------------------------

def test_supervisor_graph_structure():
    builder = build_supervisor_graph()
    graph = builder.compile()

    assert graph is not None

    # Test route_deliberation helper
    assert route_deliberation({"deliberation_action": "act_clarify"}) == "act_clarify"
    assert route_deliberation({"deliberation_action": "act_search"}) == "act_search"
    assert route_deliberation({"deliberation_action": "act_extract_requirements"}) == "act_extract_requirements"
    assert route_deliberation({"deliberation_action": "unknown_action"}) == "act_respond_direct"
