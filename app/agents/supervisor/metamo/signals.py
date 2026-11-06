"""Perceptual Signal Extractor for MetaMo Cognitive Architecture.

Extracts objective environmental cues, linguistic markers, and domain indicators
using Gemini Flash with strict Pydantic Structured Outputs (MetaMoSignals),
followed by deterministic sensory pre-calibration (calibrate_action_signals).
"""

from __future__ import annotations
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


def _heuristic_fallback_signals(
    user_query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    kg_context: str = "",
    last_cognitive_state: Optional[Dict[str, Any]] = None,
) -> MetaMoSignals:
    """Deterministic fallback estimating signals from linguistic markers when LLM is offline or in local tests."""
    q_clean = user_query.strip().lower()
    words = set(re.findall(r"\b\w+\b", q_clean))
    total_words = len(re.findall(r"\b\w+\b", user_query))

    vague_terms = {
        "something", "anything", "stuff", "thing", "fix it", "broken", "help",
        "whatever", "improve", "better", "fix", "make it", "do it", "cool"
    }
    urgent_terms = {"asap", "urgent", "emergency", "critical", "blocking", "immediately", "right now"}
    tech_terms = {"database", "postgres", "sql", "neo4j", "fastapi", "docker", "redis", "prisma", "asyncio", "mcp"}
    plan_terms = {"task", "tasks", "breakdown", "plan", "decompose", "todo", "sprint", "milestone"}
    conflict_terms = {"conflict", "conflicts", "contradiction", "discrepancy", "incompatible", "divergence"}
    search_terms = {"search", "find", "lookup", "documentation", "docs", "where", "locate", "explore"}
    greeting_terms = {"hi", "hello", "hey", "greetings", "thanks", "thank you", "ok", "okay"}

    # Ambiguity: Short, vague queries trigger high ambiguity (Anti-Premature guard)
    vague_count = sum(1 for t in vague_terms if t in q_clean)
    is_short = total_words <= 5
    amb_raw = 0.10
    if is_short and vague_count > 0:
        amb_raw += 0.65
    elif is_short and not any(t in words for t in greeting_terms):
        amb_raw += 0.55
    amb_raw += min(0.50, vague_count * 0.25)
    ambiguity = clamp_to_unit_interval(amb_raw)

    # Urgency
    urgent_count = sum(1 for t in urgent_terms if t in q_clean)
    urgency = clamp_to_unit_interval(min(1.0, urgent_count * 0.40 + user_query.count("!") * 0.15))

    # Complexity
    tech_count = sum(1 for t in tech_terms if t in words)
    has_code = "```" in user_query or "`" in user_query
    cx = clamp_to_unit_interval(min(1.0, (total_words / 80.0) * 0.35 + (tech_count * 0.15) + (0.25 if has_code else 0.0)))

    # Needs Evidence (search queries, doc references, repository inspection)
    has_search = any(t in q_clean for t in search_terms)
    evid_raw = 0.20 + (0.45 if has_search else 0.0) + (0.30 if tech_count > 0 else 0.0) + (0.25 if ("document" in q_clean or "doc" in q_clean or "file" in q_clean or "repo" in q_clean) else 0.0)
    needs_external_evidence = clamp_to_unit_interval(evid_raw)

    # Needs Task Plan
    plan_count = sum(1 for t in plan_terms if t in q_clean)
    needs_task_plan = clamp_to_unit_interval(min(1.0, plan_count * 0.45))

    # Multi source
    needs_multi_source = clamp_to_unit_interval(0.20 + (0.30 if ("compare" in q_clean or "synthesize" in q_clean) else 0.0))

    # Verify Request (explicit conflict checking or verification)
    verify_terms = {"verify", "verification", "conflict", "conflicts", "contradiction", "incompatible", "cross-check", "validate", "validating"}
    verify_request = any(t in words for t in verify_terms) or "verify" in q_clean or "cross-check" in q_clean

    # Valence
    valence = 0.0
    if any(t in words for t in ["thanks", "thank", "great", "awesome", "perfect", "good"]):
        valence = 0.60
    elif any(t in words for t in ["wrong", "bad", "error", "fail", "broken", "frustrated"]):
        valence = -0.50

    # User Expertise
    prev_exp = float(last_cognitive_state.get("user_expertise", 0.50)) if last_cognitive_state else 0.50
    if tech_count >= 2 or has_code:
        user_exp = clamp_to_unit_interval(prev_exp + 0.05)
    else:
        user_exp = prev_exp

    intent = "factual" if ("what" in words or "how" in words) else ("reflective" if ("why" in words or "explain" in words) else "mixed")

    raw = MetaMoSignals(
        cx=cx,
        ambiguity=ambiguity,
        needs_external_evidence=needs_external_evidence,
        needs_task_plan=needs_task_plan,
        needs_multi_source_integration=needs_multi_source,
        valence=valence,
        intent_type=intent,
        reflective_intent=0.6 if intent == "reflective" else 0.4,
        verify_request=verify_request,
        user_expertise=user_exp,
        urgency=urgency,
    )
    return calibrate_action_signals(raw)


async def extract_signals(
    user_query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    kg_context: str = "",
    last_cognitive_state: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> MetaMoSignals:
    """
    Extracts continuous perceptual signals using Gemini Flash with native Structured Outputs.
    Automatically applies sensory pre-calibration (calibrate_action_signals).
    Gracefully falls back to deterministic estimation if the model call fails.
    """
    model = model_name or (settings.GEMINI_MODELS[0] if getattr(settings, "GEMINI_MODELS", None) else "gemini-3.8-flash")

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
        response = await genAI.aio.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MetaMoSignals,
                temperature=0.0,
            ),
        )
        if response.text:
            signals = MetaMoSignals.model_validate_json(response.text)
            return calibrate_action_signals(signals)
    except Exception as e:
        logger.warning(f"LLM signal extraction fell back to deterministic heuristic ({e})")

    return _heuristic_fallback_signals(
        user_query=user_query,
        chat_history=chat_history,
        kg_context=kg_context,
        last_cognitive_state=last_cognitive_state,
    )
