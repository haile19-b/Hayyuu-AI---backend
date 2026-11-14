"""MetaMo Cognitive Deliberation Offline Benchmark Runner.

Executes repeatable evaluation sessions across realistic software engineering scenarios,
logging full deliberation traces out-of-band to CSV/JSONL files and reporting accuracy metrics.
"""

from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.agents.supervisor.metamo.shared import CognitiveState
from app.agents.supervisor.metamo.controller import MetaMoController
from app.agents.supervisor.metamo.homeostasis import post_turn_update
from app.agents.supervisor.metamo.eval_logger import RunLogger

# -----------------------------------------------------------------------------
# Benchmark Evaluation Sessions
# -----------------------------------------------------------------------------

BENCHMARK_SESSIONS: List[Dict[str, Any]] = [
    {
        "name": "session_onboarding_and_greetings",
        "turns": [
            {
                "query": "hello there!",
                "expected": "act_respond_direct",
                "acceptable": ["act_respond_direct"],
                "kg_context": ""
            },
            {
                "query": "Can you tell me what Hayyuu AI does?",
                "expected": "act_respond_direct",
                "acceptable": ["act_respond_direct", "act_search"],
                "kg_context": ""
            },
            {
                "query": "thank you, that is helpful",
                "expected": "act_respond_direct",
                "acceptable": ["act_respond_direct"],
                "kg_context": ""
            }
        ]
    },
    {
        "name": "session_ambiguity_and_anti_premature_guard",
        "turns": [
            {
                "query": "Fix something broken in the backend microservice architecture somehow",
                "expected": "act_clarify",
                "acceptable": ["act_clarify", "act_search"],
                "kg_context": ""
            },
            {
                "query": "Make the database faster with whatever cache stuff works",
                "expected": "act_clarify",
                "acceptable": ["act_clarify", "act_search"],
                "kg_context": ""
            }
        ]
    },
    {
        "name": "session_requirements_engineering",
        "turns": [
            {
                "query": "The system must enforce strict role-based access control (RBAC) with P0 priority for all API routes.",
                "expected": "act_extract_requirements",
                "acceptable": ["act_extract_requirements", "act_verify"],
                "kg_context": ""
            },
            {
                "query": "Define the functional requirement specification for GitHub Webhook signature validation.",
                "expected": "act_extract_requirements",
                "acceptable": ["act_extract_requirements", "act_plan_tasks"],
                "kg_context": ""
            }
        ]
    },
    {
        "name": "session_conflict_arbitration",
        "turns": [
            {
                "query": "We have an active conflict between Requirement P0-1 (serverless auto-pause) and Requirement P0-2 (sub-10ms latency).",
                "expected": "act_analyze_conflict",
                "acceptable": ["act_analyze_conflict", "act_verify"],
                "kg_context": "[Active Requirement Conflicts]\n- Conflict 101: Auto-pause vs low latency"
            }
        ]
    },
    {
        "name": "session_task_planning_and_decomposition",
        "turns": [
            {
                "query": "Break down the implementation plan for the SSE streaming endpoint into Jira tasks with acceptance criteria.",
                "expected": "act_plan_tasks",
                "acceptable": ["act_plan_tasks", "act_extract_requirements", "act_synthesize"],
                "kg_context": ""
            }
        ]
    }
]


def run_benchmark():
    parser = argparse.ArgumentParser(description="MetaMo Deliberation Benchmark Runner")
    parser.add_argument("--run-id", type=str, default=None, help="Custom identifier for the eval run")
    args = parser.parse_args()

    controller = MetaMoController()
    run_logger = RunLogger(run_id=args.run_id)

    print(f"\n=======================================================")
    print(f"🚀 MetaMo Cognitive Deliberation Benchmark")
    print(f"📁 Eval Run ID: {run_logger.run_id}")
    print(f"📊 Logging Traces to: {run_logger.csv_path}")
    print(f"=======================================================\n")

    total_turns = 0
    strict_correct_count = 0
    acceptable_correct_count = 0

    overall_t0 = time.time()

    for session in BENCHMARK_SESSIONS:
        session_name = session["name"]
        print(f"▶️ Running Scenario: {session_name}")
        state = CognitiveState()
        chat_history: List[Dict[str, Any]] = []

        for idx, turn_data in enumerate(session["turns"], start=1):
            query = turn_data["query"]
            expected = turn_data["expected"]
            acceptable = turn_data.get("acceptable", [expected])
            kg_context = turn_data.get("kg_context", "")

            t0 = time.time()
            decision = controller.deliberate(
                user_query=query,
                chat_history=chat_history,
                kg_context=kg_context,
                state=state
            )
            latency_ms = (time.time() - t0) * 1000.0

            action = decision["action"]
            is_strict = action == expected
            is_acceptable = action in acceptable

            total_turns += 1
            if is_strict:
                strict_correct_count += 1
            if is_acceptable:
                acceptable_correct_count += 1

            # Log to out-of-band CSV/JSONL
            run_logger.log_turn(
                thread_id=f"bench_{session_name}",
                turn=idx,
                query=query,
                action=action,
                style_modifier=decision["style_modifier"],
                score_top3=decision["score_top3"],
                reason=decision["reason"],
                cognitive_state=state.to_dict(),
                latency_ms=latency_ms
            )

            # Update state with post-turn homeostasis
            state_dict = post_turn_update(
                state=state.to_dict(),
                signals=decision["signals"],
                decision=decision,
                turn_success=is_acceptable
            )
            state = CognitiveState.from_dict(state_dict)

            # Append to history
            chat_history.append({"role": "USER", "content": query})
            chat_history.append({"role": "ASSISTANT", "content": f"Action: {action}"})

            status_icon = "✅" if is_strict else ("🟡" if is_acceptable else "❌")
            print(f"   Turn {idx}: {status_icon} Action='{action}' (Expected: '{expected}') | Style='{decision['style_modifier']}' | {latency_ms:.1f}ms")

        print()

    total_time = time.time() - overall_t0
    strict_acc = (strict_correct_count / total_turns) * 100.0 if total_turns > 0 else 0.0
    acceptable_acc = (acceptable_correct_count / total_turns) * 100.0 if total_turns > 0 else 0.0

    print(f"=======================================================")
    print(f"🏁 Benchmark Evaluation Completed ({total_time:.2f}s)")
    print(f"Total Evaluated Turns : {total_turns}")
    print(f"Strict Action Accuracy : {strict_acc:.1f}% ({strict_correct_count}/{total_turns})")
    print(f"Acceptable Action Acc  : {acceptable_acc:.1f}% ({acceptable_correct_count}/{total_turns})")
    print(f"Logs Written to        : {run_logger.csv_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    run_benchmark()
