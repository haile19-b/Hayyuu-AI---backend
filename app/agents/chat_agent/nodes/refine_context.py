"""Node 4: Refine Context Node for Chat Agent.

Rearranges, deduplicates, and structures raw tool findings and investigation
plan results into a high-density, factual evidence block before final synthesis.
"""

import json
import logging
from typing import Dict, Any, List

from google.genai import types

from app.agents.chat_agent.state import ChatAgentState

logger = logging.getLogger("uvicorn.error")


def _clean_json_summary(raw_str: str) -> str:
    """Extracts high-value information from raw JSON tool output."""
    try:
        data = json.loads(raw_str)
        if isinstance(data, list):
            items = []
            for item in data[:8]:
                if isinstance(item, dict):
                    # GitHub Issue / PR format
                    if "number" in item and "title" in item:
                        state_val = item.get("state", "open")
                        author = item.get("user", {}).get("login") if isinstance(item.get("user"), dict) else item.get("author", "")
                        url = item.get("html_url") or item.get("url", "")
                        items.append(f"- Issue/PR #{item['number']}: \"{item['title']}\" (State: {state_val}, Author: {author}, URL: {url})")
                    # GitHub Repo format
                    elif "full_name" in item or "fullName" in item:
                        fn = item.get("full_name") or item.get("fullName")
                        url = item.get("html_url") or item.get("htmlUrl", "")
                        branch = item.get("default_branch") or item.get("defaultBranch", "main")
                        items.append(f"- Repository: {fn} (Default Branch: {branch}, URL: {url})")
                    # Task format
                    elif "title" in item and ("priority" in item or "status" in item):
                        items.append(f"- Task: \"{item['title']}\" (Status: {item.get('status')}, Priority: {item.get('priority')})")
                    else:
                        items.append(f"- {str(item)[:200]}")
                else:
                    items.append(f"- {str(item)[:200]}")
            return "\n".join(items) if items else "No records found."
        elif isinstance(data, dict):
            # Single object
            if "number" in data and "title" in data:
                return f"- Issue/PR #{data['number']}: \"{data['title']}\" (State: {data.get('state')}, URL: {data.get('html_url')})\n  Body: {str(data.get('body', ''))[:300]}"
            return json.dumps({k: v for k, v in list(data.items())[:10] if not isinstance(v, (dict, list)) or len(str(v)) < 200}, indent=2)
    except Exception:
        pass
    return raw_str[:1500]


async def refine_context_node(state: ChatAgentState) -> Dict[str, Any]:
    """Node 4: Refines raw tool data into structured, factual evidence."""
    investigation_plan = state.get("investigation_plan", [])
    tool_results = state.get("tool_results", [])

    evidence_lines: List[str] = []

    # 1. Structure Plan Execution Steps & Summaries
    if investigation_plan:
        evidence_lines.append("### Completed Investigation Steps:")
        for step in investigation_plan:
            sid = step.get("id", "step")
            desc = step.get("description", "")
            status = step.get("status", "completed")
            summary = step.get("result_summary") or "Completed"
            evidence_lines.append(f"- [{status.upper()}] {sid}: {desc}")

    # 2. Extract and format structured tool records
    if tool_results:
        evidence_lines.append("\n### Verified Tool Findings:")
        for res in tool_results:
            t_name = res.get("tool_name", "tool")
            raw_res = res.get("result", "")
            formatted = _clean_json_summary(raw_res)
            evidence_lines.append(f"#### Tool `{t_name}` Output:\n{formatted}")

    evidence_block = "\n".join(evidence_lines) if evidence_lines else ""

    structured_evidence = f"<investigation_evidence>\n{evidence_block}\n</investigation_evidence>" if evidence_block else ""

    # Append structured evidence directly into the system instruction for final grounding
    system_instruction = state.get("system_instruction", "")
    if structured_evidence:
        system_instruction += f"\n\n{structured_evidence}\n\nCRITICAL: Ground your response strictly in the above <investigation_evidence>. Present all issues, repositories, files, and findings accurately."

    return {
        "structured_evidence": structured_evidence,
        "system_instruction": system_instruction,
        "status": "evidence_refined"
    }
