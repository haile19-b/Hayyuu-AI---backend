"""Perceptual Signal Extractor for MetaMo Cognitive Architecture.

Extracts objective environmental cues, linguistic markers, and domain indicators
using Gemini Flash with strict Pydantic Structured Outputs (MetaMoSignals),
followed by deterministic sensory pre-calibration (calibrate_action_signals).
"""

from __future__ import annotations
import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from google.genai import types

from app.core.config import genAI
from app.core.env import settings
from app.agents.supervisor.metamo.shared import (
    clamp_to_unit_interval,
    clamp_to_signed_unit_interval,
)

logger = logging.getLogger("uvicorn.error")


class MetaMoSignals(BaseModel):
    """
    Continuous sensory perceptual signals extracted from conversation context.
    Strictly defined for native Google GenAI SDK structured outputs.
    """
    cx: float = Field(
        default=0.3,
        description="Query complexity from 0.0 (simple greeting/routine factual question) to 1.0 (deep systems engineering/distributed systems)"
    )
    ambiguity: float = Field(
        default=0.2,
        description="Ambiguity or underspecification level from 0.0 (exact, clear constraints) to 1.0 (unclear, missing critical context)"
    )
    needs_external_evidence: float = Field(
        default=0.3,
        description="Degree to which codebase, uploaded documents, or GitHub repository state must be inspected (0.0 to 1.0)"
    )
    needs_task_plan: float = Field(
        default=0.2,
        description="Degree to which implementation breakdown, sprint tasks, or multi-step execution is needed (0.0 to 1.0)"
    )
    needs_multi_source_integration: float = Field(
        default=0.3,
        description="Multi-source integration requirement across multiple documents or tools (0.0 to 1.0)"
    )
    valence: float = Field(
        default=0.0,
        description="Affective user sentiment appraisal from -1.0 (frustrated, critical, correcting) to +1.0 (satisfied, praise, resolution)"
    )
    intent_type: str = Field(
        default="mixed",
        description="Conversational intent category: 'factual' | 'reflective' | 'mixed'"
    )
    reflective_intent: float = Field(
        default=0.5,
        description="Reflective reasoning depth requirement vs instant lookup (0.0 to 1.0)"
    )
    verify_request: bool = Field(
        default=False,
        description="Whether explicit verification, conflict check, or consistency audit was requested"
    )
    user_expertise: float = Field(
        default=0.5,
        description="Estimated user technical expertise from 0.0 (novice/beginner) to 1.0 (staff/principal engineer)"
    )
    urgency: float = Field(
        default=0.0,
        description="Urgency, critical blocking state, or deadline priority level (0.0 to 1.0)"
    )


def calibrate_action_signals(signals: MetaMoSignals) -> MetaMoSignals:
    """
    Deterministic sensory pre-calibration (from MetaMo prototype parser.py).
    Prevents decomposition hijacking when evidence or multi-source integration dominates.
    """
    evid = clamp_to_unit_interval(signals.needs_external_evidence)
    plan = clamp_to_unit_interval(signals.needs_task_plan)
    multi = clamp_to_unit_interval(signals.needs_multi_source_integration)
    amb = clamp_to_unit_interval(signals.ambiguity)
    refl = clamp_to_unit_interval(signals.reflective_intent)
    intent = signals.intent_type.lower() if signals.intent_type in {"factual", "reflective", "mixed"} else "mixed"

    # Keep planning signal conservative when evidence/integration dominates and ambiguity is not extreme
    if evid >= 0.75 and plan >= 0.65 and amb <= 0.70:
        plan *= 0.70
    if multi >= 0.70 and 0.60 <= plan < 0.78 and amb <= 0.65:
        plan *= 0.75
    if evid >= 0.85 and multi >= 0.75 and plan >= 0.60 and amb <= 0.55:
        plan = min(plan, 0.55)

    # Factual low-reflection contexts should not over-activate decomposition unless explicitly needed
    if intent == "factual" and refl <= 0.50 and evid >= 0.70 and plan >= 0.55:
        plan *= 0.75

    calibrated = signals.model_copy()
    calibrated.cx = clamp_to_unit_interval(signals.cx)
    calibrated.ambiguity = clamp_to_unit_interval(signals.ambiguity)
    calibrated.needs_external_evidence = clamp_to_unit_interval(evid)
    calibrated.needs_task_plan = clamp_to_unit_interval(plan)
    calibrated.needs_multi_source_integration = clamp_to_unit_interval(multi)
    calibrated.valence = clamp_to_signed_unit_interval(signals.valence)
    calibrated.reflective_intent = clamp_to_unit_interval(refl)
    calibrated.user_expertise = clamp_to_unit_interval(signals.user_expertise)
    calibrated.urgency = clamp_to_unit_interval(signals.urgency)
    calibrated.intent_type = intent
    return calibrated


async def extract_signals(
    user_query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    kg_context: str = "",
    last_cognitive_state: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> MetaMoSignals:
    """
    Extracts continuous perceptual signals strictly using Gemini with native Structured Outputs.
    Iterates across candidate Gemini models to ensure resilient LLM extraction.
    Automatically applies sensory pre-calibration (calibrate_action_signals).
    """
    from app.core.ai_cascade import generate_content_cascade, get_candidate_models

    # Build transcript context snippet
    history_snippet = ""
    if chat_history:
        recent = chat_history[-6:]
        lines = []
        for msg in recent:
            role = msg.get("role", "USER")
            content = msg.get("content", "")
            lines.append(f"{role}: {content[:200]}")
        history_snippet = "Recent Conversation History:\n" + "\n".join(lines) + "\n\n"

    kg_snippet = f"Domain / Graph Context:\n{kg_context[:500]}\n\n" if kg_context else ""

    prompt = (
        f"{history_snippet}{kg_snippet}"
        f"User Query:\n{user_query}\n\n"
        "Analyze the user's turn and output continuous cognitive indicators matching the MetaMoSignals schema."
    )

    try:
        response = await generate_content_cascade(
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MetaMoSignals,
                temperature=0.0,
            ),
            candidate_models=get_candidate_models(model_name),
            timeout_seconds=10.0,
        )
        if response and response.text:
            signals = MetaMoSignals.model_validate_json(response.text)
            return calibrate_action_signals(signals)
    except Exception as e:
        logger.error(f"[MetaMo Signals] All candidate models failed: {e}. Using baseline calibrated signals.")

    prev_exp = float(last_cognitive_state.get("user_expertise", 0.50)) if last_cognitive_state else 0.50
    return calibrate_action_signals(MetaMoSignals(user_expertise=prev_exp))
