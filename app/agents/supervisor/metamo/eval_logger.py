"""Developer Evaluation Telemetry Logger (RunLogger).

Decoupled, zero-database-bloat evaluation logger for developer benchmark testing.
Writes comprehensive cognitive deliberation traces exclusively to local CSV and JSONL files.
"""

from __future__ import annotations
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("uvicorn.error")

CSV_HEADERS = [
    "run_id",
    "timestamp",
    "thread_id",
    "turn",
    "query",
    "action",
    "style_modifier",
    "score_top3",
    "reason",
    "urgency",
    "arousal",
    "conflict_tension",
    "failure_wariness",
    "securing",
    "topic_familiarity",
    "valence",
    "user_expertise",
    "clarify_first",
    "build_knowledge",
    "extract_requirements",
    "resolve_conflict",
    "ground_facts",
    "propose_tasks",
    "verify_consistency",
    "premature_action",
    "hallucinated_assumptions",
    "latency_ms"
]


class RunLogger:
    """Thread-safe file-based evaluation logger for cognitive deliberation runs."""

    def __init__(self, run_id: Optional[str] = None, base_dir: Optional[Path] = None):
        if not base_dir:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "logs" / "eval"
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"run_{timestamp_str}"
        self.run_dir = base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.run_dir / "turns.csv"
        self.jsonl_path = self.run_dir / "turns.jsonl"
        self.meta_path = self.run_dir / "run_meta.json"

        # Initialize metadata
        if not self.meta_path.exists():
            with self.meta_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "run_id": self.run_id,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0.0"
                }, f, indent=2)

        # Initialize CSV header if newly created
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()

    def log_turn(
        self,
        thread_id: str,
        turn: int,
        query: str,
        action: str,
        style_modifier: str,
        score_top3: str,
        reason: str,
        cognitive_state: Dict[str, Any],
        latency_ms: float = 0.0
    ) -> None:
        """Appends a turn trace to both turns.csv and turns.jsonl files."""
        try:
            modulators = cognitive_state.get("modulators", {})
            goals = cognitive_state.get("goals", {})
            anti_goals = cognitive_state.get("anti_goals", {})

            record = {
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "thread_id": thread_id,
                "turn": turn,
                "query": query[:200].replace("\n", " "),
                "action": action,
                "style_modifier": style_modifier,
                "score_top3": score_top3,
                "reason": reason[:300].replace("\n", " "),
                "urgency": round(float(modulators.get("urgency", 0.0)), 3),
                "arousal": round(float(modulators.get("arousal", 0.0)), 3),
                "conflict_tension": round(float(modulators.get("conflict_tension", 0.0)), 3),
                "failure_wariness": round(float(modulators.get("failure_wariness", 0.0)), 3),
                "securing": round(float(modulators.get("securing", 0.0)), 3),
                "topic_familiarity": round(float(modulators.get("topic_familiarity", 0.0)), 3),
                "valence": round(float(cognitive_state.get("valence", 0.0)), 3),
                "user_expertise": round(float(cognitive_state.get("user_expertise", 0.5)), 3),
                "clarify_first": round(float(goals.get("clarify_first", 0.0)), 3),
                "build_knowledge": round(float(goals.get("build_knowledge", 0.0)), 3),
                "extract_requirements": round(float(goals.get("extract_requirements", 0.0)), 3),
                "resolve_conflict": round(float(goals.get("resolve_conflict", 0.0)), 3),
                "ground_facts": round(float(goals.get("ground_facts", 0.0)), 3),
                "propose_tasks": round(float(goals.get("propose_tasks", 0.0)), 3),
                "verify_consistency": round(float(goals.get("verify_consistency", 0.0)), 3),
                "premature_action": round(float(anti_goals.get("premature_action", 0.0)), 3),
                "hallucinated_assumptions": round(float(anti_goals.get("hallucinated_assumptions", 0.0)), 3),
                "latency_ms": round(latency_ms, 2)
            }

            # 1. Write CSV
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(record)

            # 2. Write JSONL
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.warning(f"Failed to write evaluation telemetry log: {e}")
