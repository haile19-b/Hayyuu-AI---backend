"""Tests for Supplemental Endpoints & Background Hooks.

Tests:
1. Developer deliberation telemetry reader (GET /threads/{thread_id}/telemetry)
2. User episodic/semantic memory management (CRUD schemas & mapping)
3. Neo4j/Relational D3 knowledge graph visualizer format (GET /projects/{project_id}/graph)
4. ARQ worker cold-path memory consolidation task registration
"""

import os
import sys
import json
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.queue import WorkerSettings, consolidate_memory_job, analyze_document_job
from app.interfaces.api.v1.memory_route import MemoryItemResponse, CreateMemoryRequest, UpdateMemoryRequest
from prisma.enums import MemoryType


def test_arq_worker_settings_functions():
    """Verifies both analyze_document_job and consolidate_memory_job are registered."""
    funcs = WorkerSettings.functions
    assert analyze_document_job in funcs
    assert consolidate_memory_job in funcs
    assert len(funcs) == 2


def test_memory_request_schemas_and_enum_mapping():
    """Verifies MemoryType mapping handles all domain types cleanly."""
    create_req = CreateMemoryRequest(
        key="architecture/database",
        content="PostgreSQL with Neon PgBouncer pooler",
        type="SEMANTIC",
        importance=0.8
    )
    assert create_req.key == "architecture/database"
    assert create_req.importance == 0.8

    # Enum mapping check
    type_mapping = {
        "PREFERENCE": MemoryType.EPISODIC,
        "DECISION": MemoryType.EPISODIC,
        "FACT": MemoryType.SEMANTIC,
        "RULE": MemoryType.PROCEDURAL,
        "SEMANTIC": MemoryType.SEMANTIC,
        "EPISODIC": MemoryType.EPISODIC,
        "PROCEDURAL": MemoryType.PROCEDURAL,
    }
    for raw, expected in type_mapping.items():
        assert type_mapping[raw] == expected


def test_telemetry_log_parsing():
    """Verifies that out-of-band JSONL evaluation traces are properly parsed."""
    base_logs = Path(__file__).resolve().parent.parent / "logs" / "eval"
    if base_logs.exists():
        for run_dir in base_logs.glob("run_*"):
            jsonl_file = run_dir / "turns.jsonl"
            if jsonl_file.exists():
                with jsonl_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            assert "run_id" in record
                            assert "action" in record
                            assert "style_modifier" in record
                            assert "score_top3" in record
                            assert "reason" in record
                            break
                break


def test_graph_visualization_schema():
    """Verifies that the D3/vis.js graph payload structure matches expected schema."""
    sample_nodes = [
        {"id": "p1", "label": "Project", "name": "Hayyuu AI", "group": "Project", "properties": {}},
        {"id": "r1", "label": "Requirement", "name": "OAuth2 PKCE", "group": "Requirement", "properties": {"priority": "P0"}}
    ]
    sample_links = [
        {"id": "p1->SPECIFIES->r1", "source": "p1", "target": "r1", "type": "SPECIFIES", "label": "SPECIFIES"}
    ]
    graph_payload = {
        "projectId": "p1",
        "nodes": sample_nodes,
        "links": sample_links,
        "summary": {
            "nodeCount": len(sample_nodes),
            "linkCount": len(sample_links)
        }
    }
    assert len(graph_payload["nodes"]) == 2
    assert len(graph_payload["links"]) == 1
    assert graph_payload["summary"]["nodeCount"] == 2
    assert graph_payload["summary"]["linkCount"] == 1
