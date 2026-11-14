"""MetaMo Homeostatic Stabilization and Post-Turn Drive Modulation.

Implements contractive mapping around homeostatic setpoints (PSI theory), bounding drift,
exponential moving average blending, hard safety floors (THETA_SAFE),
and signed affective valence updates across conversational turns.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from app.agents.supervisor.metamo.shared import (
    clamp_to_unit_interval,
    clamp_to_signed_unit_interval,
    clamp_range,
    GOAL_KEYS,
    MODULATOR_KEYS,
    ANTI_GOAL_KEYS,
    DEFAULT_GOALS,
    DEFAULT_MODULATORS,
    DEFAULT_ANTI_GOALS,
    DEFAULT_COGNITIVE_CENTERS,
    MODULATOR_BOUNDS,
    GOAL_DEFAULT_BOUNDS,
    GOAL_ALPHA,
    ANTI_GOAL_ALPHA,
    VALENCE_ALPHA,
    THETA_SAFE,
    HOMEOSTASIS_ETA,
    HOMEOSTASIS_ALPHA_NEAR,
)


def _blend(current: float, target: float, alpha: float) -> float:
    """Exponential moving average blending: (1 - alpha)*current + alpha*target."""
    return clamp_to_unit_interval((1.0 - alpha) * current + alpha * target)


def _near_boundary(value: float, lo: float, hi: float, eta: float) -> bool:
    """Checks whether a value is within eta distance of lower or upper bound."""
    return value <= (lo + eta) or value >= (hi - eta)


def _apply_scope_contractivity(
    container: Dict[str, float],
    keys: Tuple[str, ...],
    bounds: Tuple[float, float] | Dict[str, Tuple[float, float]],
    centers: Dict[str, float],
    eta: float,
    alpha_near: float,
    trigger_prefix: str,
    trigger_keys: List[str],
) -> None:
    for key in keys:
        if key not in container:
            continue
        if isinstance(bounds, dict):
            lo, hi = bounds.get(key, (0.0, 1.0))
        else:
            lo, hi = bounds
        current = float(container.get(key, centers.get(key, 0.5)))
        if _near_boundary(current, lo, hi, eta):
            center = float(centers.get(key, 0.5))
            updated = (1.0 - alpha_near) * current + alpha_near * center
            container[key] = clamp_range(updated, lo, hi)
            trigger_keys.append(f"{trigger_prefix}.{key}")
        else:
            container[key] = clamp_range(current, lo, hi)


def apply_homeostatic_contractivity(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies homeostatic contractivity to keep state variables within stable operating envelopes.
    Pulls boundary-drifting variables gently toward their setpoints, and enforces
    non-negotiable hard safety floors (over_safety, over_honesty >= THETA_SAFE).
    """
    goals = state.setdefault("goals", dict(DEFAULT_GOALS))
    modulators = state.setdefault("modulators", dict(DEFAULT_MODULATORS))
    anti_goals = state.setdefault("anti_goals", dict(DEFAULT_ANTI_GOALS))

    trigger_keys: List[str] = []

    # 1. Contract goals
    _apply_scope_contractivity(
        container=goals,
        keys=GOAL_KEYS,
        bounds=GOAL_DEFAULT_BOUNDS,
        centers=DEFAULT_GOALS,
        eta=HOMEOSTASIS_ETA,
        alpha_near=HOMEOSTASIS_ALPHA_NEAR,
        trigger_prefix="goals",
        trigger_keys=trigger_keys,
    )

    # Enforce hard safety floors
    if "over_safety" in goals:
        goals["over_safety"] = max(THETA_SAFE, goals["over_safety"])
    if "over_honesty" in goals:
        goals["over_honesty"] = max(THETA_SAFE, goals["over_honesty"])

    # 2. Contract modulators
    _apply_scope_contractivity(
        container=modulators,
        keys=MODULATOR_KEYS,
        bounds=MODULATOR_BOUNDS,
        centers=DEFAULT_MODULATORS,
        eta=HOMEOSTASIS_ETA,
        alpha_near=HOMEOSTASIS_ALPHA_NEAR,
        trigger_prefix="modulators",
        trigger_keys=trigger_keys,
    )

    # 3. Contract anti-goals
    _apply_scope_contractivity(
        container=anti_goals,
        keys=ANTI_GOAL_KEYS,
        bounds=GOAL_DEFAULT_BOUNDS,
        centers=DEFAULT_ANTI_GOALS,
        eta=HOMEOSTASIS_ETA,
        alpha_near=HOMEOSTASIS_ALPHA_NEAR,
        trigger_prefix="anti_goals",
        trigger_keys=trigger_keys,
    )

    debug = {
        "enabled": True,
        "mode": "near_boundary" if trigger_keys else "interior",
        "trigger_keys": trigger_keys,
        "trigger_count": len(trigger_keys),
    }
    state["homeostasis_debug"] = debug
    return debug


def compute_goal_targets(signals: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, float]:
    """Calculates instantaneous target values for the 14 PSI goals (adapted from MetaMo core/state/goals.py)."""
    cx = float(signals.get("cx", 0.3))
    ambiguity = float(signals.get("ambiguity", 0.2))
    urgent = float(signals.get("urgency", 0.0)) >= 0.5
    expertise = float(signals.get("user_expertise", 0.5))
    reflective = float(signals.get("reflective_intent", 0.5))
    action = decision.get("action", "act_respond_direct")

    targets: Dict[str, float] = {
        "efficiency": 0.45 + (0.35 if urgent else 0.0) + 0.20 * (1.0 - cx),
        "accuracy": 0.45 + 0.45 * cx + 0.10 * ambiguity + (0.05 if not urgent else 0.0),
        "success_moderate": 0.45 + 0.30 * cx + 0.25 * ambiguity - (0.08 if urgent else 0.0),
        "knowledge": 0.35 + 0.50 * cx + 0.20 * ambiguity + 0.10 * expertise,
        "novelty": 0.30 + 0.35 * cx + 0.20 * reflective + 0.10 * ambiguity - (0.05 if urgent else 0.0),
        "success_breakthrough": 0.30 + 0.45 * cx + 0.25 * ambiguity + 0.20 * reflective - (0.08 if urgent else 0.0),
        "coherence": 0.45 + 0.25 * (1.0 - ambiguity) + (0.10 if not urgent else 0.0),
        "originality": 0.30 + 0.40 * cx + 0.25 * reflective + 0.10 * ambiguity - (0.08 if urgent else 0.0),
        "social": 0.35 + 0.30 * ambiguity + 0.20 * (1.0 - expertise) + (0.10 if urgent else 0.0),
        "help_short": 0.35 + 0.35 * (1.0 - cx) + 0.20 * (1.0 - ambiguity) + (0.10 if urgent else 0.0),
        "help_long": 0.30 + 0.55 * cx + 0.15 * (1.0 - ambiguity),
        "over_beneficial": 0.50 + 0.25 * (1.0 - ambiguity),
        "over_safety": 0.55 + 0.30 * cx,
        "over_honesty": 0.55 + 0.25 * ambiguity,
        # Domain goals
        "clarify_first": 0.45 + 0.45 * ambiguity,
        "extract_requirements": 0.35 + 0.50 * float(signals.get("needs_external_evidence", 0.3)),
        "resolve_conflict": 0.40 + 0.50 * (1.0 if signals.get("verify_request") else 0.0),
        "propose_tasks": 0.35 + 0.45 * float(signals.get("needs_task_plan", 0.2)),
        "verify_consistency": 0.45 + 0.40 * (1.0 if signals.get("verify_request") else 0.0),
    }

    # Action-specific target modulations
    if action == "act_search":
        targets["accuracy"] += 0.05
        targets["knowledge"] += 0.06
        targets["novelty"] += 0.06
    elif action == "act_verify":
        targets["accuracy"] += 0.06
        targets["coherence"] += 0.06
        targets["over_safety"] += 0.05
        targets["over_honesty"] += 0.06
        targets["verify_consistency"] += 0.20
    elif action == "act_clarify":
        targets["clarify_first"] += 0.20
        targets["social"] += 0.05
    elif action == "act_plan_tasks":
        targets["propose_tasks"] += 0.20
        targets["efficiency"] += 0.08
    elif action == "act_extract_requirements":
        targets["extract_requirements"] += 0.20
        targets["accuracy"] += 0.05

    return {k: clamp_to_unit_interval(v) for k, v in targets.items()}


def compute_anti_goal_targets(signals: Dict[str, Any], goals: Dict[str, float]) -> Dict[str, float]:
    """Calculates instantaneous targets for anti-goals (unwanted cognitive behaviors)."""
    cx = float(signals.get("cx", 0.3))
    ambiguity = float(signals.get("ambiguity", 0.2))
    urgency = float(signals.get("urgency", 0.0))

    premature_target = 0.20 + 0.45 * cx + 0.35 * ambiguity - 0.20 * (1.0 - urgency)
    over_delegation_target = 0.20 + 0.35 * (1.0 - cx) + 0.25 * urgency
    hallucinate_target = 0.20 + 0.45 * ambiguity + 0.25 * (1.0 - goals.get("accuracy", 0.70))
    rabbit_hole_target = 0.18 + 0.40 * cx + 0.30 * (1.0 - urgency) - 0.20 * ambiguity

    return {
        "premature": clamp_to_unit_interval(premature_target),
        "redundant": clamp_to_unit_interval(over_delegation_target),
        "hallucinate": clamp_to_unit_interval(hallucinate_target),
        "rabbit_hole": clamp_to_unit_interval(rabbit_hole_target),
    }


def post_turn_update(
    state: Dict[str, Any],
    signals: Dict[str, Any],
    decision: Dict[str, Any],
    turn_success: bool = True
) -> Dict[str, Any]:
    """
    Executes the two-phase MetaMo post-turn update:
    Phase 1: Modulator adjustment based on discourse and success/failure.
    Phase 2: Signed affective valence update with dynamic failure damping.
    Phase 3: Goal and anti-goal exponential blending with homeostatic contractivity.
    """
    goals = state.setdefault("goals", dict(DEFAULT_GOALS))
    modulators = state.setdefault("modulators", dict(DEFAULT_MODULATORS))
    anti_goals = state.setdefault("anti_goals", dict(DEFAULT_ANTI_GOALS))

    cx = float(signals.get("cx", 0.3))
    ambiguity = float(signals.get("ambiguity", 0.2))
    valence_sig = float(signals.get("valence", 0.0))

    # Phase 1: Modulator adjustment
    # Technical dialogue increases topic familiarity
    if cx >= 0.40:
        modulators["topic_familiarity"] = clamp_to_unit_interval(modulators.get("topic_familiarity", 0.50) + 0.05)

    # Resolution vs failure wariness
    if turn_success:
        modulators["arousal"] = clamp_to_unit_interval(modulators.get("arousal", 0.40) - 0.06)
        modulators["failure_wariness"] = clamp_to_unit_interval(modulators.get("failure_wariness", 0.10) - 0.04)
        failure_sig = 0.0
    else:
        modulators["arousal"] = clamp_to_unit_interval(modulators.get("arousal", 0.40) + 0.12)
        modulators["failure_wariness"] = clamp_to_unit_interval(modulators.get("failure_wariness", 0.10) + 0.10)
        modulators["securing"] = clamp_to_unit_interval(modulators.get("securing", 0.30) + 0.10)
        failure_sig = 0.70

    if signals.get("verify_request"):
        modulators["conflict_tension"] = clamp_to_unit_interval(modulators.get("conflict_tension", 0.20) + 0.15)
        modulators["securing"] = clamp_to_unit_interval(modulators.get("securing", 0.30) + 0.10)
    else:
        modulators["conflict_tension"] = clamp_to_unit_interval(modulators.get("conflict_tension", 0.20) - 0.04)

    # Phase 2: Signed Valence Update (MetaMo formula)
    valence_target = clamp_to_signed_unit_interval(valence_sig - 0.10 * failure_sig)
    current_valence = float(state.get("valence", 0.0))
    updated_valence = clamp_to_signed_unit_interval(
        (1.0 - VALENCE_ALPHA) * current_valence + VALENCE_ALPHA * valence_target
    )
    state["valence"] = updated_valence

    # Phase 3: Goal and Anti-Goal EMA Blending
    target_goals = compute_goal_targets(signals, decision)
    for g_key in GOAL_KEYS:
        if g_key in target_goals:
            goals[g_key] = _blend(goals.get(g_key, DEFAULT_GOALS.get(g_key, 0.5)), target_goals[g_key], GOAL_ALPHA)

    target_anti = compute_anti_goal_targets(signals, goals)
    for a_key in ANTI_GOAL_KEYS:
        if a_key in target_anti:
            anti_goals[a_key] = _blend(anti_goals.get(a_key, DEFAULT_ANTI_GOALS.get(a_key, 0.3)), target_anti[a_key], ANTI_GOAL_ALPHA)

    # Phase 4: Contractivity stabilization
    apply_homeostatic_contractivity(state)

    state["turn_count"] = int(state.get("turn_count", 0)) + 1
    state["last_action"] = decision.get("action")
    state["last_style"] = decision.get("style_modifier")
    return state
