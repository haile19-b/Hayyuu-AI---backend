"""Curated Multi-Turn Evaluation Sessions for Hayyuu AI Benchmark Runner.

Defines realistic software engineering sessions testing MetaMo cognitive arbitration:
1. session_onboarding: Initial project greeting, grounding query, synthesis.
2. session_ambiguity: Vague user queries triggering Anti-Premature guard to force clarification.
3. session_requirements: Structured user specification extraction and inspection.
4. session_conflict: Architectural conflicting requirements triggering verification guard.
5. session_task_planning: Engineering task decomposition and sprint planning.
"""

from typing import List, Dict, Any

SESSIONS: List[Dict[str, Any]] = [
    {
        "name": "session_onboarding",
        "description": "Multi-turn project onboarding and architecture orientation.",
        "queries": [
            "Hi, I am starting a new healthcare patient portal project. What capabilities does Hayyuu AI provide?",
            "Can you search our project documentation to see if we have HIPAA compliance requirements?",
            "Please synthesize an architectural overview combining our backend API, database, and auth services."
        ],
        "expected_actions": [
            "act_respond_direct",
            "act_search",
            "act_synthesize"
        ],
        "acceptable_actions": [
            ["act_clarify", "act_respond_direct"],
            ["act_search", "act_respond_direct"],
            ["act_synthesize", "act_respond_direct"]
        ]
    },
    {
        "name": "session_ambiguity",
        "description": "Underspecified queries designed to trigger the Anti-Premature guard.",
        "queries": [
            "Make it better.",
            "I need to optimize the database query latency for user retrieval from 500ms down to 50ms.",
            "Something is broken with the system."
        ],
        "expected_actions": [
            "act_clarify",
            "act_respond_direct",
            "act_clarify"
        ],
        "acceptable_actions": [
            ["act_clarify"],
            ["act_respond_direct", "act_search"],
            ["act_clarify"]
        ]
    },
    {
        "name": "session_requirements",
        "description": "Extracting structured requirements from user stories and cross-referencing.",
        "queries": [
            "Here is our user requirement: Doctors must be able to view patient lab results within 2 seconds of upload, and export them as signed PDF reports.",
            "Search existing requirements to check if there are any other document export rules.",
            "Summarize the verified requirement status for this sprint."
        ],
        "expected_actions": [
            "act_extract_requirements",
            "act_search",
            "act_respond_direct"
        ],
        "acceptable_actions": [
            ["act_extract_requirements", "act_decompose"],
            ["act_search", "act_respond_direct"],
            ["act_respond_direct", "act_synthesize"]
        ]
    },
    {
        "name": "session_conflict",
        "description": "Conflicting technical requirements triggering Verification guard.",
        "queries": [
            "We have a conflict: Requirement REQ-1 specifies PostgreSQL for all patient records, but Requirement REQ-5 specifies MongoDB for audit logs. Please verify this conflict.",
            "How can we resolve this architectural divergence while maintaining ACID guarantees?",
            "Verify whether our current authentication flow complies with the resolved database schema."
        ],
        "expected_actions": [
            "act_verify",
            "act_respond_direct",
            "act_verify"
        ],
        "acceptable_actions": [
            ["act_verify", "act_analyze_conflict"],
            ["act_respond_direct", "act_synthesize"],
            ["act_verify", "act_analyze_conflict", "act_search"]
        ]
    },
    {
        "name": "session_task_planning",
        "description": "Decomposing complex modules into actionable engineering tasks.",
        "queries": [
            "Break down the patient appointment booking module into detailed engineering sprint tasks with priority and estimates.",
            "Search for dependencies between the booking API task and notification delivery task.",
            "Provide the final implementation roadmap for these tasks."
        ],
        "expected_actions": [
            "act_plan_tasks",
            "act_search",
            "act_respond_direct"
        ],
        "acceptable_actions": [
            ["act_plan_tasks", "act_decompose"],
            ["act_search", "act_verify"],
            ["act_respond_direct", "act_synthesize"]
        ]
    }
]
