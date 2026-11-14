"""MetaMo Shared Definitions, Clamping Helpers, and Baseline Configurations.

Based on PSI Theory of Cognitive Architecture and MetaMo Deliberation Engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Clamping Utilities (Strict Mathematical Interval Bounding)
# -----------------------------------------------------------------------------

def clamp_to_unit_interval(val: float) -> float:
    """Clamps a numeric value strictly to the standard unit interval [0.0, 1.0]."""
    try:
        f = float(val)
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f
    except (TypeError, ValueError):
        return 0.0


def clamp_to_signed_unit_interval(val: float) -> float:
    """Clamps a signed numeric value strictly to the signed unit interval [-1.0, 1.0]."""
    try:
        f = float(val)
        if f < -1.0:
            return -1.0
        if f > 1.0:
            return 1.0
        return f
    except (TypeError, ValueError):
        return 0.0


def clamp_range(val: float, lo: float, hi: float) -> float:
    """Clamps a value to the specified [lo, hi] bounds."""
    try:
        f = float(val)
        if f < lo:
            return lo
        if f > hi:
            return hi
        return f
    except (TypeError, ValueError):
        return lo


# -----------------------------------------------------------------------------
# MetaMo Cognitive Dimensions
# -----------------------------------------------------------------------------

GOAL_KEYS: Tuple[str, ...] = (
    "efficiency",
    "accuracy",
    "success_moderate",
    "knowledge",
    "novelty",
    "success_breakthrough",
    "coherence",
    "originality",
    "social",
    "help_short",
    "help_long",
    "over_beneficial",
    "over_safety",
    "over_honesty",
    # Domain-specific goals
    "clarify_first",
    "extract_requirements",
    "resolve_conflict",
    "propose_tasks",
    "verify_consistency",
)

MODULATOR_KEYS: Tuple[str, ...] = (
    "threshold",
    "arousal",
    "securing",
    "risk_aversion",
    "failure_wariness",
    "topic_familiarity",
    "resolution",
    "user_expertise",
    "urgency",
    "conflict_tension",
    "systemic_load",
    "creativity",
)

ANTI_GOAL_KEYS: Tuple[str, ...] = (
    "hallucinate",
    "redundant",
    "rabbit_hole",
    "premature",
)

ACTION_KEYS: Tuple[str, ...] = (
    "act_respond_direct",
    "act_clarify",
    "act_search",
    "act_verify",
    "act_plan_tasks",
    "act_extract_requirements",
    "act_analyze_conflict",
    "act_build_knowledge",
    "act_synthesize",
)

# -----------------------------------------------------------------------------
# Default Homeostatic Baselines & Bounds (MetaMo Setpoints)
# -----------------------------------------------------------------------------

DEFAULT_GOALS: Dict[str, float] = {
    "efficiency": 0.60,
    "accuracy": 0.70,
    "success_moderate": 0.62,
    "knowledge": 0.52,
    "novelty": 0.46,
    "success_breakthrough": 0.44,
    "coherence": 0.58,
    "originality": 0.48,
    "social": 0.50,
    "help_short": 0.55,
    "help_long": 0.45,
    "over_beneficial": 0.60,
    "over_safety": 0.65,
    "over_honesty": 0.65,
    "clarify_first": 0.55,
    "extract_requirements": 0.52,
    "resolve_conflict": 0.50,
    "propose_tasks": 0.48,
    "verify_consistency": 0.58,
}

DEFAULT_MODULATORS: Dict[str, float] = {
    "threshold": 0.30,
    "arousal": 0.40,
    "securing": 0.30,
    "risk_aversion": 0.40,
    "failure_wariness": 0.10,
    "topic_familiarity": 0.50,
    "resolution": 0.40,
    "user_expertise": 0.50,
    "urgency": 0.25,
    "conflict_tension": 0.20,
    "systemic_load": 0.30,
    "creativity": 0.45,
}

DEFAULT_ANTI_GOALS: Dict[str, float] = {
    "hallucinate": 0.35,
    "redundant": 0.30,
    "rabbit_hole": 0.28,
    "premature": 0.30,
}

DEFAULT_VALENCE: float = 0.0  # Signed [-1.0, 1.0]

DEFAULT_COGNITIVE_CENTERS: Dict[str, float] = {
    **DEFAULT_GOALS,
    **DEFAULT_MODULATORS,
    **DEFAULT_ANTI_GOALS,
}

MODULATOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "threshold": (0.20, 0.95),
    "arousal": (0.10, 0.90),
    "securing": (0.00, 1.00),
    "risk_aversion": (0.00, 1.00),
    "failure_wariness": (0.00, 1.00),
    "topic_familiarity": (0.00, 1.00),
    "resolution": (0.00, 1.00),
    "user_expertise": (0.00, 1.00),
    "urgency": (0.05, 0.95),
    "conflict_tension": (0.05, 0.95),
    "systemic_load": (0.05, 0.95),
    "creativity": (0.10, 0.90),
}

GOAL_DEFAULT_BOUNDS: Tuple[float, float] = (0.00, 1.00)

# Hyperparameters
GOAL_ALPHA: float = 0.18
ANTI_GOAL_ALPHA: float = 0.16
VALENCE_ALPHA: float = 0.20
THETA_SAFE: float = 0.55
HOMEOSTASIS_ETA: float = 0.05
HOMEOSTASIS_ALPHA_NEAR: float = 0.10


# -----------------------------------------------------------------------------
# State Representation
# -----------------------------------------------------------------------------

@dataclass
class CognitiveState:
    """Persisted thread cognitive state representing homeostatic memory."""
    turn_count: int = 0
    goals: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GOALS))
    modulators: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MODULATORS))
    anti_goals: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ANTI_GOALS))
    valence: float = DEFAULT_VALENCE
    user_expertise: float = 0.50
    last_action: Optional[str] = None
    last_style: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CognitiveState":
        if not data:
            return cls()
        return cls(
            turn_count=int(data.get("turn_count", 0)),
            goals={k: clamp_to_unit_interval(float(data.get("goals", {}).get(k, DEFAULT_GOALS.get(k, 0.5)))) for k in GOAL_KEYS},
            modulators={k: clamp_to_unit_interval(float(data.get("modulators", {}).get(k, DEFAULT_MODULATORS.get(k, 0.5)))) for k in MODULATOR_KEYS},
            anti_goals={k: clamp_to_unit_interval(float(data.get("anti_goals", {}).get(k, DEFAULT_ANTI_GOALS.get(k, 0.3)))) for k in ANTI_GOAL_KEYS},
            valence=clamp_to_signed_unit_interval(float(data.get("valence", DEFAULT_VALENCE))),
            user_expertise=clamp_to_unit_interval(float(data.get("user_expertise", 0.50))),
            last_action=data.get("last_action"),
            last_style=data.get("last_style"),
        )
