"""MetaMo Cognitive Deliberation Controller.

Evaluates 14 drives, 12 modulators, 4 anti-goals, and environmental signals
to choose optimal agent routing actions, fast-lane overrides, and output styles.
Produces rich developer evaluation telemetry while guaranteeing zero database bloat.
"""

from __future__ import annotations
import math
import logging
from typing import Dict, Any, Tuple, List, Optional
from pydantic import BaseModel

from app.agents.supervisor.metamo.shared import (
    CognitiveState,
    clamp_to_unit_interval,
    clamp_to_signed_unit_interval,
    DEFAULT_GOALS,
    DEFAULT_MODULATORS,
    DEFAULT_ANTI_GOALS,
)
from app.agents.supervisor.metamo.signals import (
    MetaMoSignals,
    extract_signals,
    calibrate_action_signals,
)
from app.agents.supervisor.metamo.style import determine_respond_style

logger = logging.getLogger("uvicorn.error")


def load_or_initialize_cognitive_state(thread_record: Any) -> CognitiveState:
    """Retrieves persisted cognitive state for existing conversation or initializes Turn-1 defaults."""
    if thread_record is None:
        return CognitiveState()
    if isinstance(thread_record, CognitiveState):
        return thread_record
    if isinstance(thread_record, dict):
        return CognitiveState.from_dict(thread_record.get("cognitiveState", thread_record))
    if hasattr(thread_record, "cognitiveState") and thread_record.cognitiveState:
        return CognitiveState.from_dict(thread_record.cognitiveState)
    return CognitiveState()


def compute_cold_start_blend(
    running_state: Dict[str, float],
    signals: MetaMoSignals,
    turn_count: int,
    horizon: int = 2,
    strength: float = 0.70
) -> Dict[str, float]:
    """Blends immediate sensory context during turns 0-1 to eliminate initial sluggishness."""
    effective = dict(running_state)
    if turn_count < horizon:
        cold_phase = (horizon - turn_count) / horizon
        cold_weight = clamp_to_unit_interval(strength * cold_phase)

        arousal_target = 0.35 * signals.cx + 0.30 * signals.urgency
        effective["arousal"] = clamp_to_unit_interval(
            (1.0 - cold_weight) * effective.get("arousal", 0.40) + cold_weight * arousal_target
        )
        effective["user_expertise"] = clamp_to_unit_interval(
            (1.0 - cold_weight) * effective.get("user_expertise", 0.50) + cold_weight * signals.user_expertise
        )
        effective["topic_familiarity"] = clamp_to_unit_interval(
            (1.0 - cold_weight) * effective.get("topic_familiarity", 0.50)
            + cold_weight * (1.0 - signals.needs_external_evidence)
        )
    return effective


class MetaMoController:
    """Deliberative cognitive routing and decision engine."""

    def __init__(self):
        pass

    async def deliberate(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        kg_context: str = "",
        state: CognitiveState | Dict[str, Any] | None = None,
        precomputed_signals: Optional[MetaMoSignals] = None,
    ) -> Dict[str, Any]:
        """
        Executes an asynchronous deliberation cycle.
        Returns:
            {
                "action": str (selected action key),
                "score_top3": str (formatted top 3 action scores),
                "scores": dict[str, float] (full score map),
                "reason": str (deliberative explanation for developer telemetry),
                "style_modifier": str,
                "signals": dict[str, Any],
                "fast_lane": bool,
                "updated_drives": dict[str, float],
            }
        """
        cog_state = load_or_initialize_cognitive_state(state)

        # 1. Perceptual Signal Extraction via Gemini Flash Structured Output
        if precomputed_signals is not None:
            signals = calibrate_action_signals(precomputed_signals)
        else:
            signals = await extract_signals(
                user_query=user_query,
                chat_history=chat_history,
                kg_context=kg_context,
                last_cognitive_state=cog_state.to_dict(),
            )

        modulators = cog_state.modulators
        turn_count = cog_state.turn_count

        # 2. Cold-Start Blend on Turns 0-1
        effective_drives = compute_cold_start_blend(modulators, signals, turn_count)

        # 3. Dynamic Response Style Shaper (MetaMo style.py math)
        style_modifier = determine_respond_style(signals, effective_drives)

        # 4. Composite Indicators
        familiarity = effective_drives.get("topic_familiarity", 0.50)
        failure_wariness = effective_drives.get("failure_wariness", 0.10)
        threshold = effective_drives.get("threshold", 0.30)
        valence = cog_state.valence

        confidence = clamp_to_unit_interval(
            0.55 * familiarity + 0.25 * (1.0 - signals.ambiguity) + 0.20 * (1.0 - signals.cx)
        )
        low_confidence = clamp_to_unit_interval(1.0 - confidence)
        answerability = clamp_to_unit_interval((1.0 - signals.ambiguity) * (1.0 - threshold) * familiarity)

        # 5. Base Candidate Action Scoring
        scores: Dict[str, float] = {
            "act_respond_direct": (
                0.50
                + 0.35 * (1.0 - signals.ambiguity)
                - 0.35 * failure_wariness
                + 0.10 * valence
                + 0.30 * answerability
                - 0.20 * signals.cx
            ),
            "act_clarify": (
                0.20
                + 0.90 * signals.ambiguity
                - 0.20 * effective_drives.get("user_expertise", 0.50)
                - 0.08 * valence
            ),
            "act_search": (
                0.30
                + 0.60 * signals.needs_external_evidence
                + 0.20 * signals.cx
                - 0.15 * signals.urgency
            ),
            "act_verify": (
                0.25
                + 0.65 * threshold
                + 0.75 * low_confidence
                - 0.14 * valence
                + (0.50 if signals.verify_request else 0.0)
            ),
            "act_plan_tasks": (
                0.25
                + 0.50 * signals.needs_task_plan
                + 0.30 * signals.cx
                - 0.25 * signals.ambiguity
            ),
            "act_extract_requirements": (
                0.25
                + 0.45 * signals.needs_external_evidence
                + 0.30 * signals.cx
                - 0.20 * signals.ambiguity
            ),
            "act_analyze_conflict": (
                0.20
                + 0.60 * effective_drives.get("conflict_tension", 0.20)
                + (0.50 if signals.verify_request else 0.0)
            ),
            "act_synthesize": (
                0.20
                + 0.50 * signals.cx
                + 0.30 * signals.needs_multi_source_integration
                + 0.08 * valence
            ),
        }

        # Context-specific bonuses
        q_lower = user_query.lower()
        if "conflict" in q_lower or "[active requirement conflicts]" in kg_context.lower():
            scores["act_analyze_conflict"] += 0.55
        if "requirement" in q_lower or "spec" in q_lower or "criteria" in q_lower:
            scores["act_extract_requirements"] += 0.45
        if "task" in q_lower or "sprint" in q_lower or "plan" in q_lower or "decompose" in q_lower:
            scores["act_plan_tasks"] += 0.40

        # Multi-source integration guard: Lock out act_synthesize unless multi-source signal >= 0.65
        if signals.needs_multi_source_integration < 0.65:
            scores["act_synthesize"] = -1e9

        fast_lane = False
        reason = "Standard deliberative scoring across cognitive candidate actions."

        # ---------------------------------------------------------------------
        # 6. Guard Arbitration & Fast-Lane Routing
        # ---------------------------------------------------------------------
        ambiguity_threshold = 0.55

        # Fast-Lane 1: Instant Familiar Response
        if (
            signals.cx <= 0.30
            and signals.ambiguity <= 0.30
            and threshold <= 0.25
            and failure_wariness <= 0.25
            and familiarity >= 0.70
            and low_confidence <= 0.35
            and not signals.verify_request
        ):
            scores["act_verify"] = -1e9
            scores["act_search"] = -1e9
            scores["act_clarify"] = -1e9
            scores["act_respond_direct"] += 0.60
            reason = "Fast-Lane 1 triggered: Low-risk familiar query directly routed to respond."
            fast_lane = True

        # Fast-Lane 2: Routine Short-Help Response
        elif (
            signals.cx <= 0.45
            and signals.ambiguity <= 0.35
            and threshold <= 0.20
            and familiarity >= 0.80
            and not signals.verify_request
        ):
            scores["act_clarify"] = -1e9
            scores["act_search"] = -1e9
            scores["act_respond_direct"] += 0.40
            reason = "Fast-Lane 2 triggered: High-familiarity routine response bypass."
            fast_lane = True

        # Anti-Premature Guard: Force clarification when query ambiguity is high
        elif signals.ambiguity >= ambiguity_threshold:
            scores["act_plan_tasks"] = -1e9
            scores["act_respond_direct"] = -1e9
            scores["act_clarify"] += 0.60
            reason = (
                f"Clarification forced: Ambiguity ({signals.ambiguity:.2f}) exceeded threshold "
                f"({ambiguity_threshold:.2f}); Anti-Premature guard locked out execution."
            )

        # Verification Guard
        elif signals.verify_request or failure_wariness >= 0.65:
            scores["act_verify"] += 0.60
            reason = f"Verification forced: Explicit verify request or elevated failure wariness ({failure_wariness:.2f})."

        # Task Planning Guard
        elif signals.needs_task_plan >= 0.65 and signals.cx >= 0.50:
            scores["act_plan_tasks"] += 0.45
            reason = (
                f"Task decomposition selected: Task plan requirement ({signals.needs_task_plan:.2f}) "
                f"and complexity ({signals.cx:.2f}) exceed thresholds."
            )

        # Evidence Gathering Guard
        elif signals.needs_external_evidence >= 0.60:
            scores["act_search"] += 0.40
            reason = f"Search selected: Evidence gathering required ({signals.needs_external_evidence:.2f})."

        # 7. Winner Selection & Telemetry Formatting
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner_action, winner_score = sorted_candidates[0]

        top3 = sorted_candidates[:3]
        top3_formatted = ", ".join([f"{act} ({sc:.3f})" for act, sc in top3])

        return {
            "action": winner_action,
            "style_modifier": style_modifier,
            "score_top3": top3_formatted,
            "scores": scores,
            "reason": reason,
            "signals": signals.model_dump(),
            "fast_lane": fast_lane,
            "updated_drives": effective_drives,
        }

    def deliberate_sync(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        kg_context: str = "",
        state: CognitiveState | Dict[str, Any] | None = None,
        precomputed_signals: Optional[MetaMoSignals] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for offline benchmark harness runner.py."""
        import asyncio
        return asyncio.run(
            self.deliberate(
                user_query=user_query,
                chat_history=chat_history,
                kg_context=kg_context,
                state=state,
                precomputed_signals=precomputed_signals,
            )
        )
