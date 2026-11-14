"""MetaMo Style Modifier Mapping.

Selects dynamic behavioral output framing (concise, cautious, thorough, tutorial, exploratory)
using continuous mathematical scoring adapted from MetaMo core/decision/style.py.
"""

from __future__ import annotations
from typing import Dict, Any
from app.agents.supervisor.metamo.signals import MetaMoSignals


def determine_respond_style(signals: MetaMoSignals, state_drives: Dict[str, float]) -> str:
    """
    Calculates optimal response style modifier adapted from MetaMo core/decision/style.py.
    Returns one of:
      - 'style_concise': Direct, punchy, minimal fluff for urgent or simple requests.
      - 'style_thorough': In-depth, structured, architectural analysis with exhaustive context.
      - 'style_cautious': Highly measured, defensive, highlighting risks/conflicts.
      - 'style_tutorial': Educational, explanatory, guiding beginner/novice users.
      - 'style_exploratory': Creative, ideating, open-ended architectural possibilities.
    """
    urgency = float(signals.urgency)
    complexity = float(signals.cx)
    ambiguity = float(signals.ambiguity)
    user_expertise = float(signals.user_expertise)
    threshold = float(state_drives.get("threshold", 0.30))
    failure_wariness = float(state_drives.get("failure_wariness", 0.10))
    resolution = float(state_drives.get("resolution", 0.40))
    creativity = float(state_drives.get("creativity", 0.45))
    verify_request = bool(signals.verify_request)

    style_scores = {
        "style_concise": 0.40 + 0.70 * urgency + 0.20 * (1.0 - complexity) + 0.15 * (1.0 - ambiguity),
        "style_thorough": 0.35 + 0.55 * complexity + 0.45 * resolution + 0.15 * ambiguity - 0.35 * urgency,
        "style_cautious": 0.35 + 0.55 * threshold + 0.30 * failure_wariness + (0.25 if verify_request else 0.0),
        "style_tutorial": 0.35 + 0.70 * (1.0 - user_expertise) + 0.20 * ambiguity + 0.10 * complexity,
        "style_exploratory": 0.30 + 0.55 * creativity - 0.35 * threshold,
    }
    return max(style_scores, key=style_scores.get)


def determine_style(modulators: Dict[str, float], selected_action: str, user_expertise: float = 0.5) -> str:
    """Backward-compatible helper mapping directly from modulators and action."""
    urgency = float(modulators.get("urgency", 0.25))
    failure_wariness = float(modulators.get("failure_wariness", 0.20))
    conflict_tension = float(modulators.get("conflict_tension", 0.20))

    if urgency >= 0.65:
        return "style_concise"
    if failure_wariness >= 0.55 or conflict_tension >= 0.60 or selected_action in ("act_verify", "act_analyze_conflict"):
        return "style_cautious"
    if user_expertise <= 0.35:
        return "style_tutorial"
    if selected_action in ("act_plan_tasks", "act_extract_requirements", "act_synthesize"):
        return "style_thorough"
    return "style_concise" if urgency >= 0.40 else "style_thorough"
