"""Tests for Phase 4: Long-Term Agentic Memory Subsystem.

Validates:
1. Native Pydantic structured output models (MemoryInsightType, MemoryItem, MemoryConsolidationOutput).
2. Deterministic sensory and heuristic fallback memory extraction.
3. In-band hot-path prompt formatting (<project_memory> XML block).
4. ARQ worker memory consolidation task registration and enqueueing.
5. REST API request/response schemas, enum mapping, and importance interval clamping.
6. Context window budgeting integration with memory context injection.
7. Native Google GenAI SDK structured output consolidation execution.
"""

import sys
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.agents.memory.consolidator import (
    MemoryInsightType,
    MemoryItem,
    MemoryConsolidationOutput,
    _heuristic_fallback_extraction,
    consolidate_thread_memories,
)
from app.agents.memory.retriever import format_memories_for_prompt
from app.agents.supervisor.context_budget import assemble_context_with_budget
from app.core.queue import WorkerSettings, consolidate_memory_job, enqueue_memory_consolidation
from app.interfaces.api.v1.memory_route import (
    MemoryItemResponse,
    CreateMemoryRequest,
    UpdateMemoryRequest,
)
from prisma.enums import MemoryType
from app.agents.supervisor.metamo.shared import clamp_to_unit_interval


# =============================================================================
# 1. Structured Output Schema Validation
# =============================================================================

def test_memory_structured_output_schemas():
    """Verifies that native Pydantic memory schemas validate correctly."""
    item = MemoryItem(
        key="architecture/database",
        type=MemoryInsightType.DECISION,
        content="Decided to use PostgreSQL with Neon PgBouncer connection pooler.",
        importance=0.9
    )
    assert item.key == "architecture/database"
    assert item.type == MemoryInsightType.DECISION
    assert item.importance == 0.9

    output = MemoryConsolidationOutput(memories=[item])
    assert len(output.memories) == 1

    # Test serialization to JSON and parsing back
    json_str = output.model_dump_json()
    parsed = MemoryConsolidationOutput.model_validate_json(json_str)
    assert len(parsed.memories) == 1
    assert parsed.memories[0].key == "architecture/database"
    assert parsed.memories[0].type == MemoryInsightType.DECISION


# =============================================================================
# 2. Deterministic Heuristic Fallback Extraction Tests
# =============================================================================

def test_heuristic_fallback_extraction():
    """Verifies fallback extraction from conversation dialogue when LLM is offline."""
    messages = [
        {"role": "USER", "content": "Let's decide on the stack."},
        {"role": "ASSISTANT", "content": "We decided to use FastAPI for the backend."},
        {"role": "USER", "content": "I prefer TypeScript and strict linting."},
        {"role": "ASSISTANT", "content": "Rule: All database mutations must ensure transactional rollbacks."},
    ]

    extracted = _heuristic_fallback_extraction(messages)
    assert len(extracted) >= 2

    keys = [item.key for item in extracted]
    types = [item.type for item in extracted]

    assert any("architecture" in k for k in keys)
    assert MemoryInsightType.DECISION in types or MemoryInsightType.PREFERENCE in types


# =============================================================================
# 3. Prompt Formatting Helper Tests (<project_memory>)
# =============================================================================

def test_format_memories_for_prompt():
    """Verifies XML-tagged <project_memory> prompt formatting."""
    # Empty case
    assert format_memories_for_prompt([]) == ""

    # Populated case
    memories = [
        {
            "key": "architecture/database",
            "type": "DECISION",
            "content": "PostgreSQL Neon pooler",
            "importance": 0.85
        },
        {
            "key": "preferences/code_style",
            "type": "PREFERENCE",
            "content": "Python with strict type annotations",
            "importance": 0.75
        }
    ]

    formatted = format_memories_for_prompt(memories)
    assert "<project_memory>" in formatted
    assert "</project_memory>" in formatted
    assert "[DECISION] architecture/database: PostgreSQL Neon pooler (importance: 0.85)" in formatted
    assert "[PREFERENCE] preferences/code_style: Python with strict type annotations (importance: 0.75)" in formatted


# =============================================================================
# 4. ARQ Worker Settings & Background Task Registration Tests
# =============================================================================

def test_arq_worker_consolidation_job():
    """Verifies that consolidate_memory_job is registered in WorkerSettings."""
    assert consolidate_memory_job in WorkerSettings.functions
    assert WorkerSettings.job_timeout >= 600


# =============================================================================
# 5. REST API Schemas & Importance Clamping Tests
# =============================================================================

def test_memory_route_schemas_and_clamping():
    """Verifies API request schemas and interval clamping."""
    create_req = CreateMemoryRequest(
        key="security/auth",
        content="JWT with refresh tokens stored in HTTP-only cookies",
        type="RULE",
        importance=1.5  # Out of range, should clamp
    )
    clamped_importance = clamp_to_unit_interval(create_req.importance)
    assert clamped_importance == 1.0

    update_req = UpdateMemoryRequest(
        content="Updated auth rule",
        importance=-0.2  # Negative, should clamp
    )
    assert clamp_to_unit_interval(update_req.importance) == 0.0

    # Verify Enum mapping
    assert MemoryType["EPISODIC"] == MemoryType.EPISODIC
    assert MemoryType["SEMANTIC"] == MemoryType.SEMANTIC
    assert MemoryType["PROCEDURAL"] == MemoryType.PROCEDURAL


# =============================================================================
# 6. Context Window Budgeting with Memory Context Injection Tests
# =============================================================================

def test_context_budget_memory_injection():
    """Verifies assemble_context_with_budget incorporates memory_context."""
    async def _run():
        memories = [
            {
                "key": "architecture/auth",
                "type": "DECISION",
                "content": "Use OAuth2 PKCE flow for mobile authentication",
                "importance": 0.90
            }
        ]

        mock_prisma = MagicMock()
        mock_prisma.chatmessage.find_many = AsyncMock(return_value=[])
        mock_prisma.project.find_unique = AsyncMock(return_value=None)
        mock_prisma.requirementconflict.find_many = AsyncMock(return_value=[])
        mock_prisma.task.find_many = AsyncMock(return_value=[])
        mock_prisma.requirement.find_many = AsyncMock(return_value=[])

        with patch("app.agents.supervisor.context_budget.prisma", mock_prisma):
            ctx = await assemble_context_with_budget(
                thread_id="test_thread",
                project_id="test_project",
                user_query="How should mobile clients authenticate?",
                semantic_search=False,
                memories=memories,
            )

            assert "memory_context" in ctx
            assert "<project_memory>" in ctx["memory_context"]
            assert "architecture/auth" in ctx["memory_context"]
            assert ctx["total_estimated_tokens"] > 0

    asyncio.run(_run())


# =============================================================================
# 7. Native Structured Output Consolidation Execution Tests
# =============================================================================

def test_native_structured_output_consolidation():
    """Verifies consolidate_thread_memories parses native structured output from Gemini Flash."""
    async def _run():
        fake_llm_response = MagicMock()
        fake_llm_response.text = json.dumps({
            "memories": [
                {
                    "key": "architecture/cache",
                    "type": "DECISION",
                    "content": "Use Redis for hot session caching and ARQ queues.",
                    "importance": 0.85
                },
                {
                    "key": "preferences/language",
                    "type": "PREFERENCE",
                    "content": "Maintain Python 3.11 compatibility.",
                    "importance": 0.70
                }
            ]
        })

        messages = [
            {"role": "USER", "content": "Should we use Redis for caching?"},
            {"role": "ASSISTANT", "content": "Yes, let's use Redis for hot session caching and ARQ queues."},
            {"role": "USER", "content": "Please keep Python 3.11 compatibility in mind."},
            {"role": "ASSISTANT", "content": "Understood, all code will target Python 3.11."},
        ]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=fake_llm_response)

        mock_store = AsyncMock()
        mock_prisma = MagicMock()
        mock_prisma.memoryrecord.find_first = AsyncMock(return_value=None)
        mock_created = MagicMock()
        mock_created.id = "mem_123"
        mock_created.key = "architecture/cache"
        mock_created.type = MemoryType.EPISODIC
        mock_prisma.memoryrecord.create = AsyncMock(return_value=mock_created)

        with patch("app.agents.memory.consolidator.genAI", mock_client):
            with patch("app.agents.memory.consolidator.get_memory_store", AsyncMock(return_value=mock_store)):
                with patch("app.agents.memory.consolidator.prisma", mock_prisma):
                    saved = await consolidate_thread_memories(
                        user_id="user_test_01",
                        project_id="proj_test_01",
                        thread_id="thread_test_01",
                        messages=messages,
                    )

                    # Verify mock LLM was called with native structured output configuration
                    assert mock_client.aio.models.generate_content.called
                    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
                    config = call_kwargs.get("config")
                    assert config is not None
                    assert config.response_mime_type == "application/json"
                    assert config.response_schema == MemoryConsolidationOutput

                    # Verify store and prisma persistence were invoked
                    assert mock_store.aput.called
                    assert mock_prisma.memoryrecord.create.called
                    assert len(saved) >= 1

    asyncio.run(_run())
