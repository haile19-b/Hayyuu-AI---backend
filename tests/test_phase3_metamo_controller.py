"""Tests for Phase 3: MetaMo Cognitive Deliberation Controller, Homeostasis, and Telemetry.

Validates:
1. Clamping functions (unit interval [0,1], signed unit interval [-1,1]).
2. MetaMoSignals schema and deterministic sensory calibration (calibrate_action_signals).
3. CognitiveState initialization, serialization, and deserialization.
4. Cold-start blending on Turns 0-1.
5. Dynamic response style shaper (determine_respond_style and determine_style).
6. Deliberation controller guard arbitration (Fast-Lane 1 & 2, Anti-Premature, Verification, Task Planning, Evidence).
7. Post-turn homeostasis, valence blending, boundary contractivity, and hard safety floors (THETA_SAFE = 0.55).
8. Out-of-band evaluation telemetry logger (RunLogger) writing to CSV and JSONL with zero database bloat.
"""

import pytest
import os
import shutil
import tempfile
from pathlib import Path

from app.agents.supervisor.metamo.shared import (
    clamp_to_unit_interval,
    clamp_to_signed_unit_interval,
    clamp_range,
    CognitiveState,
    DEFAULT_GOALS,
    DEFAULT_MODULATORS,
    DEFAULT_ANTI_GOALS,
    THETA_SAFE,
)
from app.agents.supervisor.metamo.signals import (
    MetaMoSignals,
    calibrate_action_signals,
    _heuristic_fallback_signals,
)
from app.agents.supervisor.metamo.style import (
    determine_respond_style,
    determine_style,
)
from app.agents.supervisor.metamo.controller import (
    MetaMoController,
    compute_cold_start_blend,
    load_or_initialize_cognitive_state,
)
from app.agents.supervisor.metamo.homeostasis import (
    post_turn_update,
    apply_homeostatic_contractivity,
    compute_goal_targets,
    compute_anti_goal_targets,
)
from app.agents.supervisor.metamo.eval_logger import RunLogger, CSV_HEADERS


# =============================================================================
# 1. Clamping Helpers Tests
# =============================================================================

def test_clamping_helpers():
    """Verifies strict mathematical interval bounding."""
    # Unit interval [0.0, 1.0]
    assert clamp_to_unit_interval(-0.5) == 0.0
    assert clamp_to_unit_interval(1.5) == 1.0
    assert clamp_to_unit_interval(0.42) == 0.42
    assert clamp_to_unit_interval("invalid") == 0.0

    # Signed unit interval [-1.0, 1.0]
    assert clamp_to_signed_unit_interval(-1.5) == -1.0
    assert clamp_to_signed_unit_interval(1.5) == 1.0
    assert clamp_to_signed_unit_interval(-0.75) == -0.75
    assert clamp_to_signed_unit_interval(0.85) == 0.85
    assert clamp_to_signed_unit_interval("bad") == 0.0

    # Custom range
    assert clamp_range(15.0, 0.0, 10.0) == 10.0
    assert clamp_range(-5.0, 0.0, 10.0) == 0.0
    assert clamp_range(5.0, 0.0, 10.0) == 5.0


# =============================================================================
# 2. Perceptual Signals & Pre-Calibration Tests
# =============================================================================

def test_signals_validation_and_precalibration():
    """Verifies Pydantic schema validation and decomposition calibration."""
    # Test valid model
    sig = MetaMoSignals(
        cx=0.7,
        ambiguity=0.3,
        needs_external_evidence=0.85,
        needs_task_plan=0.75,
        needs_multi_source_integration=0.80,
        valence=0.2,
        intent_type="mixed",
        reflective_intent=0.6,
        verify_request=False,
        user_expertise=0.8,
        urgency=0.4,
    )
    assert sig.cx == 0.7

    # Test calibrate_action_signals:
    # When evid >= 0.85 and multi >= 0.75 and plan >= 0.60 and amb <= 0.55 -> plan capped at 0.55
    calibrated = calibrate_action_signals(sig)
    assert calibrated.needs_task_plan <= 0.55

    # Test fallback signals
    fallback = _heuristic_fallback_signals(
        user_query="Can we verify the conflicting database schemas immediately?",
        chat_history=[],
    )
    assert fallback.verify_request is True
    assert fallback.urgency > 0.0
    assert 0.0 <= fallback.cx <= 1.0


# =============================================================================
# 3. CognitiveState Serialization Tests
# =============================================================================

def test_cognitive_state_lifecycle():
    """Verifies default state initialization, serialization, and deserialization."""
    state = CognitiveState()
    assert state.turn_count == 0
    assert state.valence == 0.0
    assert state.goals["accuracy"] == DEFAULT_GOALS["accuracy"]
    assert state.modulators["arousal"] == DEFAULT_MODULATORS["arousal"]
    assert state.anti_goals["premature"] == DEFAULT_ANTI_GOALS["premature"]

    # Serialization to dict
    data = state.to_dict()
    assert isinstance(data, dict)
    assert data["turn_count"] == 0
    assert "goals" in data
    assert "modulators" in data

    # Deserialization from dict
    data["turn_count"] = 5
    data["valence"] = 0.65
    data["goals"]["accuracy"] = 0.85
    restored = CognitiveState.from_dict(data)
    assert restored.turn_count == 5
    assert restored.valence == 0.65
    assert restored.goals["accuracy"] == 0.85

    # Mock Prisma thread record loader
    class MockThread:
        cognitiveState = data

    loaded = load_or_initialize_cognitive_state(MockThread())
    assert loaded.turn_count == 5
    assert loaded.valence == 0.65


# =============================================================================
# 4. Cold-Start Blending Tests
# =============================================================================

def test_cold_start_blending():
    """Verifies immediate sensory context blending on Turns 0 and 1."""
    drives = dict(DEFAULT_MODULATORS)
    signals = MetaMoSignals(cx=0.9, urgency=0.8, user_expertise=0.9, needs_external_evidence=0.2)

    # Turn 0: Maximum cold blend
    blend_turn0 = compute_cold_start_blend(drives, signals, turn_count=0, horizon=2, strength=0.70)
    assert blend_turn0["user_expertise"] > drives["user_expertise"]
    assert blend_turn0["arousal"] > drives["arousal"]

    # Turn 1: Partial cold blend
    blend_turn1 = compute_cold_start_blend(drives, signals, turn_count=1, horizon=2, strength=0.70)
    assert blend_turn1["user_expertise"] >= drives["user_expertise"]

    # Turn 2: Cold phase ended, no modification
    blend_turn2 = compute_cold_start_blend(drives, signals, turn_count=2, horizon=2, strength=0.70)
    assert blend_turn2["user_expertise"] == drives["user_expertise"]
    assert blend_turn2["arousal"] == drives["arousal"]


# =============================================================================
# 5. Dynamic Style Shaper Tests
# =============================================================================

def test_style_shaper():
    """Verifies continuous style scoring matching MetaMo decision/style.py."""
    # Urgent query -> style_concise
    sig_urgent = MetaMoSignals(urgency=0.95, cx=0.2, ambiguity=0.1)
    style_urgent = determine_respond_style(sig_urgent, DEFAULT_MODULATORS)
    assert style_urgent == "style_concise"

    # Deep complex query with high resolution -> style_thorough
    sig_complex = MetaMoSignals(urgency=0.0, cx=0.9, ambiguity=0.2, user_expertise=0.8)
    mod_thorough = dict(DEFAULT_MODULATORS, resolution=0.85)
    style_complex = determine_respond_style(sig_complex, mod_thorough)
    assert style_complex == "style_thorough"

    # High wariness / threshold -> style_cautious
    sig_cautious = MetaMoSignals(urgency=0.1, cx=0.5, ambiguity=0.2, verify_request=True)
    mod_cautious = dict(DEFAULT_MODULATORS, threshold=0.80, failure_wariness=0.75)
    style_cautious = determine_respond_style(sig_cautious, mod_cautious)
    assert style_cautious == "style_cautious"

    # Novice user -> style_tutorial
    sig_novice = MetaMoSignals(urgency=0.1, cx=0.4, ambiguity=0.3, user_expertise=0.05)
    style_novice = determine_respond_style(sig_novice, DEFAULT_MODULATORS)
    assert style_novice == "style_tutorial"

    # Backward compatibility helper
    assert determine_style({"urgency": 0.8}, "act_respond_direct") == "style_concise"
    assert determine_style({"failure_wariness": 0.7}, "act_verify") == "style_cautious"


# =============================================================================
# 6. Guard Arbitration & Deliberation Routing Tests
# =============================================================================

@pytest.mark.asyncio
async def test_fast_lane_1_guard():
    """Verifies Fast-Lane 1 instantly routes routine low-risk familiar queries directly."""
    controller = MetaMoController()
    state = CognitiveState(
        turn_count=3,
        modulators=dict(
            DEFAULT_MODULATORS,
            topic_familiarity=0.85,
            threshold=0.20,
            failure_wariness=0.10,
        )
    )
    signals = MetaMoSignals(
        cx=0.15,
        ambiguity=0.10,
        needs_external_evidence=0.10,
        needs_task_plan=0.10,
        needs_multi_source_integration=0.10,
        verify_request=False,
    )

    decision = await controller.deliberate(
        user_query="Hello, what time is it?",
        state=state,
        precomputed_signals=signals,
    )
    assert decision["action"] == "act_respond_direct"
    assert decision["fast_lane"] is True
    assert "Fast-Lane 1" in decision["reason"]


@pytest.mark.asyncio
async def test_anti_premature_guard():
    """Verifies that high ambiguity (>= 0.55) locks out execution and forces clarification."""
    controller = MetaMoController()
    signals = MetaMoSignals(
        cx=0.5,
        ambiguity=0.75,  # High ambiguity
        needs_external_evidence=0.8,
        needs_task_plan=0.8,
    )

    decision = await controller.deliberate(
        user_query="Fix that thing in the database",
        precomputed_signals=signals,
    )
    assert decision["action"] == "act_clarify"
    assert decision["fast_lane"] is False
    assert decision["scores"]["act_respond_direct"] == -1e9
    assert decision["scores"]["act_plan_tasks"] == -1e9
    assert "Anti-Premature guard" in decision["reason"]


@pytest.mark.asyncio
async def test_verification_guard():
    """Verifies explicit verification request or high failure wariness forces act_verify."""
    controller = MetaMoController()
    signals = MetaMoSignals(
        cx=0.6,
        ambiguity=0.2,
        verify_request=True,
    )

    decision = await controller.deliberate(
        user_query="Please verify that the user migration matches the security policy",
        precomputed_signals=signals,
    )
    assert decision["action"] == "act_verify"
    assert "Verification forced" in decision["reason"]


@pytest.mark.asyncio
async def test_task_planning_guard():
    """Verifies complex multi-step planning queries route to act_plan_tasks."""
    controller = MetaMoController()
    signals = MetaMoSignals(
        cx=0.70,
        ambiguity=0.20,
        needs_task_plan=0.85,
        needs_external_evidence=0.30,
        needs_multi_source_integration=0.20,
    )

    decision = await controller.deliberate(
        user_query="Create a sprint plan to decompose the microservice refactoring into tickets",
        precomputed_signals=signals,
    )
    assert decision["action"] == "act_plan_tasks"
    assert "Task decomposition selected" in decision["reason"]


@pytest.mark.asyncio
async def test_evidence_search_guard():
    """Verifies external evidence need routes to act_search."""
    controller = MetaMoController()
    signals = MetaMoSignals(
        cx=0.55,
        ambiguity=0.25,
        needs_external_evidence=0.80,
        needs_task_plan=0.20,
    )

    decision = await controller.deliberate(
        user_query="Find the schema definition for user sessions in the backend code",
        precomputed_signals=signals,
    )
    assert decision["action"] == "act_search"


@pytest.mark.asyncio
async def test_synthesize_lockout_guard():
    """Verifies act_synthesize is locked out when needs_multi_source_integration < 0.65."""
    controller = MetaMoController()
    signals = MetaMoSignals(
        cx=0.8,
        ambiguity=0.2,
        needs_multi_source_integration=0.40,
    )

    decision = await controller.deliberate(
        user_query="Explain the system architecture",
        precomputed_signals=signals,
    )
    assert decision["scores"]["act_synthesize"] == -1e9
    assert decision["action"] != "act_synthesize"


# =============================================================================
# 7. Post-Turn Homeostasis, Valence, & Hard Safety Floor Tests
# =============================================================================

def test_homeostasis_post_turn_update():
    """Verifies two-phase post-turn drive modulation, valence EMA, and boundary contractivity."""
    state = CognitiveState().to_dict()
    initial_arousal = state["modulators"]["arousal"]
    initial_wariness = state["modulators"]["failure_wariness"]

    signals = {
        "cx": 0.6,
        "ambiguity": 0.1,
        "valence": 0.8,
        "urgency": 0.2,
        "needs_external_evidence": 0.3,
        "needs_task_plan": 0.2,
        "verify_request": False,
        "user_expertise": 0.7,
        "reflective_intent": 0.5,
    }
    decision = {"action": "act_respond_direct", "style_modifier": "style_thorough"}

    # 1. Successful turn: arousal & wariness decrease, topic familiarity increases
    updated_success = post_turn_update(state, signals, decision, turn_success=True)
    assert updated_success["turn_count"] == 1
    assert updated_success["modulators"]["arousal"] < initial_arousal
    assert updated_success["modulators"]["failure_wariness"] <= initial_wariness
    assert updated_success["modulators"]["topic_familiarity"] > DEFAULT_MODULATORS["topic_familiarity"]
    assert updated_success["valence"] > 0.0  # Positive valence blended

    # 2. Failed turn: arousal and failure wariness spike
    prev_arousal = updated_success["modulators"]["arousal"]
    prev_wariness = updated_success["modulators"]["failure_wariness"]
    updated_fail = post_turn_update(updated_success, signals, decision, turn_success=False)
    assert updated_fail["turn_count"] == 2
    assert updated_fail["modulators"]["arousal"] > prev_arousal
    assert updated_fail["modulators"]["failure_wariness"] > prev_wariness
    assert updated_fail["modulators"]["securing"] > DEFAULT_MODULATORS["securing"]


def test_homeostatic_hard_safety_floor():
    """Verifies that non-negotiable safety floor THETA_SAFE = 0.55 is strictly enforced."""
    state = CognitiveState().to_dict()

    # Artificially drag safety and honesty down to dangerous levels
    state["goals"]["over_safety"] = 0.10
    state["goals"]["over_honesty"] = 0.20

    # Apply contractivity
    apply_homeostatic_contractivity(state)

    # Floor must strictly hold
    assert state["goals"]["over_safety"] >= THETA_SAFE
    assert state["goals"]["over_honesty"] >= THETA_SAFE
    assert state["goals"]["over_safety"] >= 0.55
    assert state["goals"]["over_honesty"] >= 0.55


# =============================================================================
# 8. Evaluation Telemetry Logger Tests (Zero Database Bloat)
# =============================================================================

def test_eval_logger_zero_db_bloat():
    """Verifies RunLogger creates out-of-band CSV and JSONL files without database writes."""
    temp_dir = Path(tempfile.mkdtemp(prefix="metamo_eval_test_"))
    try:
        logger = RunLogger(run_id="test_run_123", base_dir=temp_dir)
        assert logger.csv_path.exists()
        assert logger.meta_path.exists()

        state = CognitiveState(turn_count=1).to_dict()
        logger.log_turn(
            thread_id="thread_test_001",
            turn=1,
            query="Analyze requirement conflicts",
            action="act_analyze_conflict",
            style_modifier="style_cautious",
            score_top3="act_analyze_conflict (1.250), act_verify (0.950), act_respond_direct (0.400)",
            reason="Conflict tension elevated.",
            cognitive_state=state,
            latency_ms=45.2,
        )

        assert logger.csv_path.stat().st_size > 0
        assert logger.jsonl_path.exists()
        assert logger.jsonl_path.stat().st_size > 0

        # Read CSV content
        csv_content = logger.csv_path.read_text(encoding="utf-8")
        assert "thread_test_001" in csv_content
        assert "act_analyze_conflict" in csv_content
        assert "style_cautious" in csv_content

        # Read JSONL content
        jsonl_content = logger.jsonl_path.read_text(encoding="utf-8")
        assert "thread_test_001" in jsonl_content
        assert "latency_ms" in jsonl_content

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
