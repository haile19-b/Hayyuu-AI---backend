import sys
import uuid
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.database import prisma, connect_db, disconnect_db
from app.infrastructure.mcp.server import create_mcp_server, create_sse_app
from app.agents.supervisor.context_budget import assemble_context_with_budget, estimate_tokens
from prisma.enums import MessageRole, MemoryType, TaskStatus, TaskPriority, SuggestionStatus, ConflictSeverity, ConflictStatus
from prisma import Json


def get_call_text(result) -> str:
    """Extracts text content from an MCP CallToolResult or content list."""
    if hasattr(result, "content") and result.content:
        return result.content[0].text
    if isinstance(result, list) and result:
        return result[0].text
    return str(result)


@pytest.fixture
def mcp_server():
    return create_mcp_server(name="test-phase2-mcp")


@pytest.mark.asyncio
async def test_mcp_all_expanded_tools_discovery(mcp_server):
    """Verifies that all required expanded database tools are discovered on the MCP server."""
    tools = await mcp_server.list_tools()
    tool_names = {t.name for t in tools}

    expected_phase2_tools = {
        # Chat history traversal
        "get_chat_history",
        "search_chat_history",
        # Requirement management
        "list_project_requirements",
        "get_requirement_details",
        "create_project_requirement",
        "update_requirement_status",
        # Task management
        "list_project_tasks",
        "get_task_details",
        "update_task_details",
        "assign_task_to_requirement",
        # Conflict management
        "list_project_conflicts",
        "get_conflict_details",
        "resolve_requirement_conflict",
        # Suggestions & Repositories
        "list_project_suggestions",
        "apply_ai_suggestion",
        "reject_ai_suggestion",
        "get_document_metadata",
        "list_project_repositories",
    }

    missing = expected_phase2_tools - tool_names
    assert not missing, f"Missing required Phase 2 tools: {missing}"


@pytest.mark.asyncio
async def test_mcp_sse_health_endpoint():
    """Verifies standalone SSE application health check endpoint."""
    app = create_sse_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "hayyuu-mcp"
        assert data["tools_count"] > 15


@pytest.mark.asyncio
async def test_mcp_expanded_tools_execution(mcp_server):
    """Verifies execution of all new Phase 2 tools against database entities."""
    await connect_db()
    uid = str(uuid.uuid4())[:8]

    try:
        # Create test fixtures
        user = await prisma.user.create(
            data={
                "email": f"mcp_{uid}@example.com",
                "hashedPassword": "hash",
                "fullName": f"MCP User {uid}",
                "userName": f"mcp_{uid}",
            }
        )
        project = await prisma.project.create(
            data={
                "userId": user.id,
                "name": f"MCP Test Project {uid}",
                "description": "Integration test project for MCP Phase 2",
            }
        )

        # 1. create_project_requirement
        res_req = await mcp_server.call_tool(
            "create_project_requirement",
            {
                "project_id": project.id,
                "title": "Real-Time WebSocket Ingestion",
                "description": "Must stream live telemetry at 60Hz without dropping frames.",
                "type": "NON_FUNCTIONAL",
                "priority": "P0",
            }
        )
        req_text = get_call_text(res_req)
        assert "Successfully created Requirement" in req_text
        assert "Real-Time WebSocket Ingestion" in req_text

        # Extract created requirement ID
        req_id = req_text.split("[ID: ")[1].split("]")[0]

        # 2. get_requirement_details
        res_req_det = await mcp_server.call_tool("get_requirement_details", {"requirement_id": req_id})
        req_det_text = get_call_text(res_req_det)
        assert "Real-Time WebSocket Ingestion" in req_det_text
        assert "NON_FUNCTIONAL" in req_det_text

        # 3. update_requirement_status
        res_req_stat = await mcp_server.call_tool(
            "update_requirement_status",
            {"requirement_id": req_id, "status": "DRAFT"}
        )
        assert "status successfully updated to 'DRAFT'" in get_call_text(res_req_stat)

        # 4. create a task via prisma and test get_task_details
        task = await prisma.task.create(
            data={
                "projectId": project.id,
                "title": "Configure WebSocket Gateway",
                "description": "Setup FastAPI websockets endpoint with connection tracking.",
                "priority": TaskPriority.P0,
                "status": TaskStatus.TODO,
            }
        )
        res_task_det = await mcp_server.call_tool("get_task_details", {"task_id": task.id})
        assert "Configure WebSocket Gateway" in get_call_text(res_task_det)

        # 5. assign_task_to_requirement
        res_assign = await mcp_server.call_tool(
            "assign_task_to_requirement",
            {"task_id": task.id, "requirement_id": req_id}
        )
        assert "Successfully assigned Task" in get_call_text(res_assign)

        # 6. create a second requirement and a conflict, test get_conflict_details
        req2 = await prisma.requirement.create(
            data={
                "projectId": project.id,
                "title": "Batch Ingestion Only",
                "description": "System should only ingest batch files every midnight.",
                "type": "NON_FUNCTIONAL",
                "priority": "P1",
            }
        )
        conflict = await prisma.requirementconflict.create(
            data={
                "projectId": project.id,
                "requirementId": req_id,
                "conflictingRequirementId": req2.id,
                "severity": ConflictSeverity.HIGH,
                "description": "Real-time streaming contradicts batch ingestion only constraint.",
            }
        )
        res_conf_det = await mcp_server.call_tool("get_conflict_details", {"conflict_id": conflict.id})
        assert "Requirement Conflict" in get_call_text(res_conf_det)
        assert "HIGH" in get_call_text(res_conf_det)

        # 7. resolve_requirement_conflict
        res_conf_res = await mcp_server.call_tool(
            "resolve_requirement_conflict",
            {
                "conflict_id": conflict.id,
                "resolution_notes": "Adopt hybrid ingestion: streaming for telemetry, batch for logs."
            }
        )
        assert "resolved successfully" in get_call_text(res_conf_res)

        # 8. AI Suggestions: apply_ai_suggestion and reject_ai_suggestion
        sugg1 = await prisma.aisuggestion.create(
            data={
                "projectId": project.id,
                "type": "requirement",
                "content": Json({
                    "title": "Automated Backpressure Control",
                    "description": "Drop or buffer packets when consumer queue exceeds 10,000 items."
                }),
                "status": SuggestionStatus.PENDING,
            }
        )
        res_apply = await mcp_server.call_tool("apply_ai_suggestion", {"suggestion_id": sugg1.id})
        apply_text = get_call_text(res_apply)
        assert "successfully accepted" in apply_text
        assert "Created Requirement 'Automated Backpressure Control'" in apply_text

        sugg2 = await prisma.aisuggestion.create(
            data={
                "projectId": project.id,
                "type": "task",
                "content": Json({
                    "title": "Unnecessary Legacy Migration",
                    "description": "Migrate deprecated v1 endpoints."
                }),
                "status": SuggestionStatus.PENDING,
            }
        )
        res_reject = await mcp_server.call_tool(
            "reject_ai_suggestion",
            {"suggestion_id": sugg2.id, "reason": "Legacy endpoints already decommissioned."}
        )
        assert "successfully rejected" in get_call_text(res_reject)

        # 9. get_document_metadata
        doc = await prisma.document.create(
            data={
                "projectId": project.id,
                "name": "architecture_blueprint.pdf",
                "filePath": "uploads/architecture_blueprint.pdf",
                "fileType": "pdf",
                "sizeBytes": 1048576,
            }
        )
        res_doc = await mcp_server.call_tool("get_document_metadata", {"document_id": doc.id})
        assert "architecture_blueprint.pdf" in get_call_text(res_doc)
        assert "1048576 bytes" in get_call_text(res_doc)

        # 10. search_chat_history across project
        thread = await prisma.conversationthread.create(
            data={
                "projectId": project.id,
                "userId": user.id,
                "title": "Traversal Test Thread",
            }
        )
        await prisma.chatmessage.create(
            data={
                "threadId": thread.id,
                "role": MessageRole.USER,
                "content": "What is our policy on automated backpressure control?",
            }
        )
        res_search = await mcp_server.call_tool(
            "search_chat_history",
            {
                "project_id": project.id,
                "query": "backpressure control",
                "semantic": False,
            }
        )
        assert "backpressure control" in get_call_text(res_search)

        # Cleanup fixtures
        await prisma.project.delete(where={"id": project.id})
        await prisma.user.delete(where={"id": user.id})
    finally:
        await disconnect_db()


@pytest.mark.asyncio
async def test_context_budgeting_engine():
    """Verifies tiered assembly, intent gating, and budget ceiling enforcement."""
    await connect_db()
    uid = str(uuid.uuid4())[:8]

    try:
        user = await prisma.user.create(
            data={
                "email": f"budget_{uid}@example.com",
                "hashedPassword": "hash",
                "fullName": f"Budget User {uid}",
                "userName": f"budget_{uid}",
            }
        )
        project = await prisma.project.create(
            data={
                "userId": user.id,
                "name": f"Budget Project {uid}",
                "description": "Project testing context budgeting",
            }
        )
        thread = await prisma.conversationthread.create(
            data={
                "projectId": project.id,
                "userId": user.id,
                "title": "Budgeting Thread",
            }
        )

        # Insert 10 chronological messages
        for i in range(10):
            await prisma.chatmessage.create(
                data={
                    "threadId": thread.id,
                    "role": MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                    "content": f"Message turn {i}: Detailed discussion on architectural components number {i}.",
                }
            )

        # Insert conflicting requirements
        req1 = await prisma.requirement.create(
            data={
                "projectId": project.id,
                "title": "High Throughput Processing",
                "description": "Handle 50k req/sec.",
                "priority": "P0",
            }
        )
        req2 = await prisma.requirement.create(
            data={
                "projectId": project.id,
                "title": "Strict Synchronous ACID Logging",
                "description": "Log each event synchronously to disk.",
                "priority": "P0",
            }
        )
        await prisma.requirementconflict.create(
            data={
                "projectId": project.id,
                "requirementId": req1.id,
                "conflictingRequirementId": req2.id,
                "severity": ConflictSeverity.HIGH,
                "description": "Synchronous logging bottlenecks high throughput requirement.",
            }
        )

        # Case A: Intent = act_verify -> must inject Active Conflicts
        ctx_verify = await assemble_context_with_budget(
            thread_id=thread.id,
            project_id=project.id,
            user_query="Are there any conflicts with throughput?",
            action="act_verify",
            semantic_search=False,
        )
        assert len(ctx_verify["tier1_chat_history"]) > 0
        assert "[Active Requirement Conflicts]" in ctx_verify["kg_context"]
        assert "Synchronous logging bottlenecks" in ctx_verify["kg_context"]

        # Case B: Intent = act_decompose -> must include tasks / requirements
        ctx_decompose = await assemble_context_with_budget(
            thread_id=thread.id,
            project_id=project.id,
            user_query="Please plan tasks for the sprint.",
            action="act_decompose",
            semantic_search=False,
        )
        assert len(ctx_decompose["tier1_chat_history"]) > 0
        assert "[Project Anchor: " in ctx_decompose["kg_context"]

        # Case C: Strict ceiling limit - verify trimming preserves Tier 1
        ctx_ceiling = await assemble_context_with_budget(
            thread_id=thread.id,
            project_id=project.id,
            user_query="Architectural components summary",
            total_budget_ceiling=150,  # very tight ceiling
            tier1_tokens=80,
            tier2_tokens=50,
            kg_tokens=50,
            semantic_search=False,
        )
        assert ctx_ceiling["total_estimated_tokens"] <= 200
        assert len(ctx_ceiling["tier1_chat_history"]) > 0

        # Cleanup fixtures
        await prisma.project.delete(where={"id": project.id})
        await prisma.user.delete(where={"id": user.id})
    finally:
        await disconnect_db()
