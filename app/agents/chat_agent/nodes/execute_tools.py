"""Node 3: Execute Tools Node for Chat Agent.

Evaluates and executes dynamic tool calls via Model Context Protocol (MCP),
supporting GitHub repository lookups, database queries, and knowledge graph exploration.
"""

import time
import logging
from typing import Dict, Any, List
from google.genai import types

from app.infrastructure.mcp.client_manager import MCPClientManager
from app.core.ai_cascade import generate_content_cascade
from app.agents.chat_agent.state import ChatAgentState

logger = logging.getLogger("uvicorn.error")

GITHUB_REPO_TOOLS = {
    "list_issues", "get_issue", "create_issue", "create_github_issue", "update_issue",
    "add_issue_comment", "list_pull_requests", "get_pull_request", "create_pull_request",
    "update_pull_request", "merge_pull_request", "create_pull_request_review",
    "list_branches", "get_branch", "create_branch", "list_commits", "get_commit",
    "search_code", "get_file_contents"
}


async def execute_tools_node(state: ChatAgentState) -> Dict[str, Any]:
    """Node 3: Autonomous Plan -> Do tool execution cycle."""
    investigation_plan = list(state.get("investigation_plan", []))
    pending_steps = [s for s in investigation_plan if s.get("status") == "pending"]

    # If no pending steps remain or circuit breaker reached, mark plan complete
    loop_count = state.get("tool_loop_count", 0) + 1
    if not pending_steps or loop_count > 8:
        return {
            "is_plan_complete": True,
            "tool_loop_count": loop_count,
            "status": "plan_completed"
        }

    current_step = pending_steps[0]
    current_step["status"] = "in_progress"

    mcp_mgr = MCPClientManager.get_instance()
    if not mcp_mgr or not mcp_mgr.available_tools:
        try:
            mcp_mgr = MCPClientManager()
            await mcp_mgr.connect_all()
        except Exception as conn_err:
            logger.warning(f"[ChatAgent: execute_tools] MCP on-demand connection failed: {conn_err}")
            current_step["status"] = "failed"
            current_step["result_summary"] = f"Connection error: {conn_err}"
            return {
                "investigation_plan": investigation_plan,
                "is_plan_complete": True,
                "status": "tools_bypassed"
            }

    gemini_tools = mcp_mgr.get_gemini_tools()
    if not gemini_tools:
        logger.debug("[ChatAgent: execute_tools] No MCP tools discovered; bypassing.")
        current_step["status"] = "failed"
        return {
            "investigation_plan": investigation_plan,
            "is_plan_complete": True,
            "status": "tools_bypassed"
        }

    system_instruction = state.get("system_instruction", "")
    gemini_contents = list(state.get("gemini_contents", []))
    project_id = state["project_id"]
    connected_repos = state.get("connected_repositories", [])
    default_repo = connected_repos[0]["fullName"] if connected_repos else None

    tool_results: List[Dict[str, Any]] = list(state.get("tool_results", []))
    pending_calls: List[Dict[str, Any]] = []

    # Directive for current step execution
    tool_hint = current_step.get("tool_hint")
    step_directive = (
        f"\n\nCURRENT OBJECTIVE (Step {current_step['id']}): {current_step['description']}\n"
        f"{f'Target Tool: You MUST invoke `{tool_hint}` to fulfill this step.' if tool_hint else 'Invoke the appropriate tool.'}\n"
        f"Never assume a file, credential, or URL does not exist without explicitly checking with `get_file_contents`."
    )

    try:
        tool_config = types.GenerateContentConfig(
            system_instruction=system_instruction + step_directive,
            tools=gemini_tools,
            temperature=0.0
        )
        pre_flight = await generate_content_cascade(
            contents=gemini_contents,
            config=tool_config,
            timeout_seconds=25.0
        )

        if pre_flight and pre_flight.function_calls:
            if pre_flight.candidates:
                gemini_contents.append(pre_flight.candidates[0].content)

            response_parts = []
            tool_outputs_summary = []
            for call in pre_flight.function_calls:
                t0 = time.time()
                call_args = dict(call.args) if call.args else {}

                # 1. Enforce project_id to the actual active project (prevent LLM hallucinations)
                call_args["project_id"] = project_id

                # 2. Auto-inject default repository if tool is a repo tool and repo is unspecified
                if call.name in GITHUB_REPO_TOOLS:
                    if not call_args.get("repo") and default_repo:
                        call_args["repo"] = default_repo
                    elif call_args.get("repo") and default_repo and "/" not in call_args.get("repo", ""):
                        call_args["repo"] = default_repo
                if call.name == "get_file_contents" and "path" not in call_args:
                    call_args["path"] = ""

                logger.info(f"[ChatAgent: execute_tools] Step '{current_step['id']}' executing MCP tool '{call.name}' with args {call_args}")
                tool_output = await mcp_mgr.call_tool(call.name, call_args)
                elapsed_ms = (time.time() - t0) * 1000.0

                record = {
                    "tool_name": call.name,
                    "arguments": call_args,
                    "result": str(tool_output),
                    "execution_time_ms": elapsed_ms,
                    "status": "success"
                }
                tool_results.append(record)
                pending_calls.append(record)
                tool_outputs_summary.append(str(tool_output)[:250])

                response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": str(tool_output)}
                    )
                )

            if response_parts:
                gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=response_parts
                    )
                )

            current_step["status"] = "completed"
            current_step["result_summary"] = "; ".join(tool_outputs_summary)

        elif pre_flight and pre_flight.text:
            # Model resolved step without tool call
            current_step["status"] = "completed"
            current_step["result_summary"] = pre_flight.text[:300]
        else:
            current_step["status"] = "completed"
            current_step["result_summary"] = "No tool call was required."

    except Exception as tool_err:
        logger.warning(f"[ChatAgent: execute_tools] Step '{current_step['id']}' failed: {tool_err}")
        current_step["status"] = "failed"
        current_step["result_summary"] = str(tool_err)[:200]

    remaining = [s for s in investigation_plan if s.get("status") == "pending"]
    is_complete = len(remaining) == 0 or loop_count >= 8

    return {
        "gemini_contents": gemini_contents,
        "tool_results": tool_results,
        "pending_tool_calls": pending_calls,
        "investigation_plan": investigation_plan,
        "tool_loop_count": loop_count,
        "is_plan_complete": is_complete,
        "status": "plan_completed" if is_complete else "step_completed"
    }
