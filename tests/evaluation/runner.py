"""Multi-Turn Cognitive Benchmark Runner for Hayyuu AI.

Adapted from the MetaMo-Prototype benchmark harness for automated regression
testing of the Hayyuu AI Cognitive Supervisor across multi-turn sessions.

Features:
- Evaluates decision accuracy: strict_correct (exact action) and soft_credit (acceptable actions).
- Homeostatic state continuity: tracks cognitive drives across multi-turn conversations.
- Out-of-band telemetry: writes comprehensive traces to CSV/JSONL via RunLogger (zero DB bloat).
- Generates strict_per_turn.json, strict_per_session.json, and strict_overall.json evaluation files.
"""

from __future__ import annotations
import os
import sys
import copy
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from app.agents.supervisor.metamo.shared import CognitiveState
from app.agents.supervisor.metamo.controller import MetaMoController
from app.agents.supervisor.metamo.homeostasis import post_turn_update
from app.agents.supervisor.metamo.eval_logger import RunLogger
from tests.evaluation.sessions.hayyuu_sessions import SESSIONS


class BenchmarkRunner:
    """Automated multi-turn evaluation runner for MetaMo cognitive governance."""

    def __init__(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        soft_credit: float = 0.50,
        eval_output_dir: Optional[Path] = None,
        run_id: Optional[str] = None
    ):
        self.sessions = sessions or SESSIONS
        self.soft_credit = soft_credit
        self.run_id = run_id or f"bench_{int(time.time())}"

        backend_root = Path(__file__).resolve().parent.parent.parent
        self.eval_output_dir = eval_output_dir or (backend_root / "logs" / "eval" / self.run_id)
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)

        self.run_logger = RunLogger(run_id=self.run_id, base_dir=self.eval_output_dir.parent)
        self.controller = MetaMoController()

    def compute_turn_eval(
        self,
        session_name: str,
        turn_index: int,
        query: str,
        expected_action: str,
        acceptable_actions: List[str],
        predicted_action: str,
        style_modifier: str,
        latency_ms: float
    ) -> Dict[str, Any]:
        """Calculates strict and soft scores for an individual turn."""
        strict_correct = int(predicted_action == expected_action)
        acceptable_hit = int(predicted_action in acceptable_actions and not strict_correct)
        soft_score = 1.0 if strict_correct else (self.soft_credit if acceptable_hit else 0.0)

        return {
            "session": session_name,
            "turn": turn_index,
            "query": query,
            "expected_action": expected_action,
            "acceptable_actions": acceptable_actions,
            "predicted_action": predicted_action,
            "style_modifier": style_modifier,
            "strict_correct": strict_correct,
            "acceptable_hit": acceptable_hit,
            "soft_score": soft_score,
            "latency_ms": round(latency_ms, 2)
        }

    async def run_single_turn(
        self,
        session_name: str,
        turn_index: int,
        query: str,
        expected_action: str,
        acceptable_actions: List[str],
        running_cog_state: Dict[str, Any],
        chat_history: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Executes one cognitive deliberation cycle and updates homeostatic drives."""
        t0 = time.time()

        # Execute MetaMo deliberation
        deliberation = await self.controller.deliberate(
            user_query=query,
            chat_history=chat_history,
            kg_context="",
            state=running_cog_state
        )

        latency_ms = (time.time() - t0) * 1000.0
        predicted_action = deliberation["action"]
        style_modifier = deliberation["style_modifier"]

        # Log out-of-band evaluation telemetry
        self.run_logger.log_turn(
            thread_id=f"{self.run_id}_{session_name}",
            turn=turn_index,
            query=query,
            action=predicted_action,
            style_modifier=style_modifier,
            score_top3=deliberation.get("score_top3", []),
            reason=deliberation.get("reason", ""),
            cognitive_state=running_cog_state,
            latency_ms=latency_ms
        )

        # Compute metric scores
        turn_eval = self.compute_turn_eval(
            session_name=session_name,
            turn_index=turn_index,
            query=query,
            expected_action=expected_action,
            acceptable_actions=acceptable_actions,
            predicted_action=predicted_action,
            style_modifier=style_modifier,
            latency_ms=latency_ms
        )

        # Post-turn homeostasis drive update
        updated_cog_state = post_turn_update(
            state=running_cog_state,
            signals=deliberation.get("signals", {}),
            decision={"action": predicted_action, "style_modifier": style_modifier},
            turn_success=bool(strict_correct := (predicted_action == expected_action or predicted_action in acceptable_actions))
        )

        return updated_cog_state, turn_eval

    async def run_session(self, session: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Executes all turns within a multi-turn session."""
        session_name = session["name"]
        queries = session["queries"]
        expected_actions = session["expected_actions"]
        acceptable_actions = session.get("acceptable_actions", [[] for _ in queries])

        running_cog_state = CognitiveState().to_dict()
        chat_history: List[Dict[str, Any]] = []
        turn_evals: List[Dict[str, Any]] = []

        session_strict = 0
        session_soft = 0.0

        for turn_idx, (q, exp, acc) in enumerate(zip(queries, expected_actions, acceptable_actions), start=1):
            running_cog_state, turn_eval = await self.run_single_turn(
                session_name=session_name,
                turn_index=turn_idx,
                query=q,
                expected_action=exp,
                acceptable_actions=acc,
                running_cog_state=running_cog_state,
                chat_history=chat_history
            )
            turn_evals.append(turn_eval)
            session_strict += turn_eval["strict_correct"]
            session_soft += turn_eval["soft_score"]

            # Append to turn history
            chat_history.append({"role": "user", "content": q})
            chat_history.append({"role": "assistant", "content": f"Handled via {turn_eval['predicted_action']}"})

        num_turns = len(queries)
        session_summary = {
            "session": session_name,
            "turns": num_turns,
            "strict_correct": session_strict,
            "strict_accuracy": round(session_strict / num_turns, 3) if num_turns else 0.0,
            "soft_score_sum": round(session_soft, 3),
            "soft_accuracy": round(session_soft / num_turns, 3) if num_turns else 0.0,
            "final_cognitive_state": running_cog_state
        }

        return turn_evals, session_summary

    async def run_all(self) -> Dict[str, Any]:
        """Runs the entire benchmark across all sessions and outputs evaluation artifacts."""
        all_turns: List[Dict[str, Any]] = []
        session_summaries: List[Dict[str, Any]] = []

        total_strict = 0
        total_soft = 0.0
        total_turns = 0

        for session in self.sessions:
            t_evals, s_summary = await self.run_session(session)
            all_turns.extend(t_evals)
            session_summaries.append(s_summary)

            total_strict += s_summary["strict_correct"]
            total_soft += s_summary["soft_score_sum"]
            total_turns += s_summary["turns"]

        overall_strict_acc = round(total_strict / total_turns, 3) if total_turns else 0.0
        overall_soft_acc = round(total_soft / total_turns, 3) if total_turns else 0.0

        overall_report = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_count": len(self.sessions),
            "total_turns": total_turns,
            "strict_correct_total": total_strict,
            "strict_accuracy": overall_strict_acc,
            "soft_score_total": round(total_soft, 3),
            "soft_accuracy": overall_soft_acc,
            "sessions": session_summaries
        }

        # Save evaluation JSON files
        self._save_eval_files(all_turns, session_summaries, overall_report)
        return overall_report

    def _save_eval_files(
        self,
        turn_records: List[Dict[str, Any]],
        session_records: List[Dict[str, Any]],
        overall_report: Dict[str, Any]
    ) -> None:
        """Persists evaluation reports to disk."""
        turn_file = self.eval_output_dir / "strict_per_turn.json"
        session_file = self.eval_output_dir / "strict_per_session.json"
        overall_file = self.eval_output_dir / "strict_overall.json"

        turn_file.write_text(json.dumps(turn_records, indent=2), encoding="utf-8")
        session_file.write_text(json.dumps(session_records, indent=2), encoding="utf-8")
        overall_file.write_text(json.dumps(overall_report, indent=2), encoding="utf-8")


async def main():
    """CLI Entrypoint for running multi-turn benchmark."""
    runner = BenchmarkRunner()
    print(f"🚀 Running Hayyuu AI MetaMo Multi-Turn Benchmark (Run ID: {runner.run_id})...\n")
    report = await runner.run_all()

    print("=" * 60)
    print(f"BENCHMARK COMPLETED: {report['session_count']} sessions, {report['total_turns']} turns")
    print(f"Strict Accuracy : {report['strict_accuracy'] * 100:.1f}% ({report['strict_correct_total']}/{report['total_turns']})")
    print(f"Soft Accuracy   : {report['soft_accuracy'] * 100:.1f}% (with acceptable hit credit)")
    print("=" * 60)
    for s in report["sessions"]:
        print(f" - {s['session']:25}: Strict {s['strict_accuracy']*100:.1f}% | Soft {s['soft_accuracy']*100:.1f}%")
    print("=" * 60)
    print(f"📁 Evaluation artifacts written to: {runner.eval_output_dir}\n")


if __name__ == "__main__":
    asyncio.run(main())
