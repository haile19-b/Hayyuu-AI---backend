import logging
from typing import Optional
from mcp.server import MCPServer

logger = logging.getLogger("uvicorn.error")


def register_agent_prompts(mcp: MCPServer) -> None:
    """Registers MCP Prompts on the given MCPServer instance."""

    @mcp.prompt()
    def analyze_conflicts(project_id: str, severity_filter: Optional[str] = None) -> str:
        """
        Generates a specialized prompt for analyzing requirement conflicts in a project,
        evaluating contradictory specifications, and recommending actionable resolutions.

        Args:
            project_id: The UUID of the project.
            severity_filter: Optional filter ('HIGH', 'MEDIUM', 'LOW').
        """
        sev_instruction = f"Focus particularly on conflicts with severity '{severity_filter}'." if severity_filter else "Analyze all active conflicts across all severity levels."
        return f"""You are Hayyuu AI's Lead Software Architect and Requirements Engineering Specialist.

Task:
Perform a deep conflict resolution analysis for Project '{project_id}'.
{sev_instruction}

Step-by-Step Instructions:
1. Use the 'list_project_conflicts' tool to retrieve all active conflict records for project '{project_id}'.
2. For each identified conflict:
   - Identify Requirement A and Requirement B involved in the conflict.
   - Use 'traverse_project_subgraph' with their IDs to analyze their shared entity dependencies and affected components.
   - Analyze the root cause of the contradiction (e.g. data consistency vs latency, scope mismatch, conflicting tech stacks).
   - Assess the architectural impact (blast radius) on existing developer tasks and services.
3. Propose a concrete resolution:
   - Option 1 (Compromise / Trade-off): How to reconcile both requirements with clear architectural trade-offs.
   - Option 2 (Supersede / Override): Recommend which requirement should take precedence and why.
4. Output your analysis using clear Markdown headers, referencing entity IDs ([REQ-xxx], [TASK-xxx]) for traceability.
"""

    @mcp.prompt()
    def task_breakdown_spec(project_id: str, requirement_id: str) -> str:
        """
        Generates a prompt for decomposing a specific software requirement into
        testable, modular developer tasks with priorities and acceptance criteria.

        Args:
            project_id: The UUID of the project.
            requirement_id: The UUID of the requirement to decompose.
        """
        return f"""You are Hayyuu AI's Technical Lead and Sprint Planning Assistant.

Task:
Decompose Requirement '{requirement_id}' in Project '{project_id}' into clear, actionable, and atomic developer tasks.

Step-by-Step Instructions:
1. Use 'list_project_requirements' to inspect the full title, type (FUNCTIONAL/NON_FUNCTIONAL), and priority of Requirement '{requirement_id}'.
2. Use 'traverse_project_subgraph' with entity ID '{requirement_id}' to discover related components, database models, and existing tasks.
3. Break down the implementation into atomic tasks:
   - Database / Schema Migration tasks (if applicable).
   - Backend API endpoint & service logic tasks.
   - Validation, error handling, and authorization rules.
   - Automated unit and integration testing tasks.
4. For each task, provide:
   - Title: Imperative concise action (e.g. 'Implement Redis rate-limiting middleware')
   - Description: Technical scope and implementation guidance
   - Priority: P0 (Critical path), P1 (Important), or P2 (Nice to have)
   - Acceptance Criteria: Checkbox list of verifiable outcomes
5. Once finalized, offer to invoke 'create_project_task' for each agreed-upon item.
"""

    @mcp.prompt()
    def architecture_synthesis(project_id: str) -> str:
        """
        Generates a prompt for synthesizing document extractions and Neo4j subgraphs
        into a unified architecture blueprint and system component overview.

        Args:
            project_id: The UUID of the project.
        """
        return f"""You are Hayyuu AI's Enterprise System Architect.

Task:
Synthesize a holistic Architecture Blueprint for Project '{project_id}'.

Step-by-Step Instructions:
1. Read the project summary resource 'hayyuu://projects/{project_id}/summary' to understand overall project scale.
2. Call 'get_project_graph_summary' to inspect the knowledge graph density.
3. Call 'list_project_documents' to understand the source documentation basis.
4. Call 'list_project_requirements' to extract core functional capabilities and non-functional constraints.
5. Produce a structured Architectural Blueprint containing:
   - Executive Summary
   - System Context & High-Level Architecture (C4 Model Level 1/2)
   - Core Domain Entities & Relationships
   - Key Architectural Decisions & Non-Functional Quality Attributes
   - Known Risk Areas or Active Conflicts (referencing 'list_project_conflicts')
"""

    @mcp.prompt()
    def github_pr_review_template(project_id: str, repo: str, pull_number: int) -> str:
        """
        Generates a prompt for performing an automated code review on a GitHub pull request.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            pull_number: Pull request number.
        """
        return f"""You are Hayyuu AI's Senior Code Reviewer and Quality Gatekeeper.

Task:
Review Pull Request #{pull_number} in GitHub repository '{repo}' for Project '{project_id}'.

Review Workflow:
1. Call 'get_pull_request' for repo '{repo}', pull_number {pull_number} to inspect title, description, branches, and diff statistics.
2. Read project requirements via 'list_project_requirements' to verify that the PR aligns with approved project requirements.
3. Assess the code changes against:
   - Security: Injection vulnerabilities, sensitive secrets, missing auth checks.
   - Robustness: Error handling, edge cases, null safety, resource cleanup.
   - Performance: N+1 database queries, unindexed filters, blocking calls.
   - Maintainability: Code clarity, naming conventions, docstrings.
4. Structure your review summary:
   - Overview & Verdict: 'APPROVE', 'REQUEST_CHANGES', or 'COMMENT'
   - Strengths: What was done well
   - Actionable Concerns & Recommendations (with specific line suggestions where possible)
5. When instructed, invoke 'create_pull_request_review' to submit your verdict.
"""

    @mcp.prompt()
    def github_issue_spec_template(project_id: str, repo: str, title: str) -> str:
        """
        Generates a prompt for authoring an actionable, production-ready GitHub issue specification.

        Args:
            project_id: Active project UUID.
            repo: Target repository (e.g. 'owner/repo').
            title: Candidate title of the issue.
        """
        return f"""You are Hayyuu AI's Technical Product Manager.

Task:
Author a comprehensive GitHub issue specification titled '{title}' for repository '{repo}' in Project '{project_id}'.

Guidelines:
1. Query relevant requirements via 'list_project_requirements' and tasks via 'list_project_tasks'.
2. Format the issue body in GitHub-Flavored Markdown with the following sections:
   - **Problem Statement / User Story**: Who needs this and why.
   - **Technical Context**: Affected files, architectural components, or database schemas.
   - **Proposed Implementation**: High-level technical approach.
   - **Acceptance Criteria**: Verifiable task checklist (- [ ] ...).
   - **Testing Plan**: Unit, integration, and manual verification steps.
3. Propose appropriate labels (e.g. 'enhancement', 'bug', 'p0', 'backend').
4. When confirmed, offer to call 'create_issue' to publish the issue to '{repo}'.
"""

