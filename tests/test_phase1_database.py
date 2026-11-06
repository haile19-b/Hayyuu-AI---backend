import sys
import uuid
import asyncio
import pytest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.database import (
    prisma,
    connect_db,
    disconnect_db,
    get_checkpointer,
    get_memory_store,
    get_pg_pool,
)
from prisma.enums import MessageRole, MemoryType, WorkflowStatus
from prisma import Json


@pytest.mark.asyncio
async def test_database_connection_and_pool():
    await connect_db()
    assert prisma.is_connected(), "Prisma should be connected"

    # Verify pgvector extension
    res = await prisma.query_raw("SELECT extname FROM pg_extension WHERE extname = 'vector';")
    assert len(res) == 1 and res[0]["extname"] == "vector", "pgvector extension must be active"

    # Verify HNSW index on chat_messages
    indexes = await prisma.query_raw("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'chat_messages' AND indexname = 'chat_messages_embedding_hnsw_idx';
    """)
    assert len(indexes) == 1, "chat_messages_embedding_hnsw_idx must exist on chat_messages"

    # Verify LangGraph Saver and Store instances
    checkpointer = await get_checkpointer()
    assert checkpointer is not None, "AsyncPostgresSaver checkpointer must be initialized"

    memory_store = await get_memory_store()
    assert memory_store is not None, "AsyncPostgresStore must be initialized"

    # Verify public tables in database
    tables = await prisma.query_raw("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' ORDER BY table_name;
    """)
    table_names = {t["table_name"] for t in tables}
    expected_tables = {
        "conversation_threads",
        "chat_messages",
        "memory_records",
        "agent_workflow_runs",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "store",
    }
    for t in expected_tables:
        assert t in table_names, f"Expected table '{t}' was not found in database"

    await disconnect_db()


@pytest.mark.asyncio
async def test_phase1_crud_and_referential_integrity():
    await connect_db()
    uid = str(uuid.uuid4())[:8]

    # Create test user
    user = await prisma.user.create(
        data={
            "email": f"test_{uid}@example.com",
            "hashedPassword": "test_password_hash",
            "fullName": f"Test User {uid}",
            "userName": f"tester_{uid}",
        }
    )
    assert user.id is not None

    # Create test project
    project = await prisma.project.create(
        data={
            "userId": user.id,
            "name": f"Test Project {uid}",
            "description": "Integration test project for Phase 1",
        }
    )
    assert project.id is not None

    # 1. Test ConversationThread with cognitiveState and metadata
    thread = await prisma.conversationthread.create(
        data={
            "projectId": project.id,
            "userId": user.id,
            "title": "Phase 1 Verification Thread",
            "cognitiveState": Json({"turn_count": 1, "arousal": 0.45, "valence": 0.1}),
            "metadata": Json({"client": "test_suite", "env": "integration"}),
        }
    )
    assert thread.id is not None
    assert thread.title == "Phase 1 Verification Thread"

    # 2. Test ChatMessage creation and compound index chronological retrieval
    msg1 = await prisma.chatmessage.create(
        data={
            "threadId": thread.id,
            "role": MessageRole.USER,
            "content": "Hello, how does the memory store work?",
            "cognitiveAction": "act_respond",
        }
    )
    msg2 = await prisma.chatmessage.create(
        data={
            "threadId": thread.id,
            "role": MessageRole.ASSISTANT,
            "content": "The memory store uses PostgreSQL with PGVector.",
            "cognitiveAction": "act_respond",
            "styleModifier": "style_concise",
        }
    )
    assert msg1.id is not None
    assert msg2.id is not None

    messages = await prisma.chatmessage.find_many(
        where={"threadId": thread.id},
        order={"createdAt": "asc"},
    )
    assert len(messages) == 2
    assert messages[0].content == "Hello, how does the memory store work?"
    assert messages[1].content == "The memory store uses PostgreSQL with PGVector."

    # 3. Test MemoryRecord with project relation and cascading
    memory = await prisma.memoryrecord.create(
        data={
            "userId": user.id,
            "projectId": project.id,
            "type": MemoryType.SEMANTIC,
            "key": "architecture/database",
            "content": "Neon PostgreSQL with pgvector extension.",
            "importance": 0.9,
        }
    )
    assert memory.id is not None
    assert memory.projectId == project.id

    # 4. Test AgentWorkflowRun
    workflow_run = await prisma.agentworkflowrun.create(
        data={
            "projectId": project.id,
            "threadId": thread.id,
            "workflowType": "requirements_extraction",
            "status": WorkflowStatus.COMPLETED,
            "inputPayload": Json({"prompt": "Extract features"}),
            "result": Json({"status": "ok", "count": 5}),
        }
    )
    assert workflow_run.id is not None
    assert workflow_run.status == WorkflowStatus.COMPLETED

    # 5. Test LangGraph Store item storage and retrieval
    store = await get_memory_store()
    namespace = (user.id, project.id, "test_memories")
    await store.aput(namespace, "pref_1", {"theme": "dark", "verbosity": "high"})
    stored_item = await store.aget(namespace, "pref_1")
    assert stored_item is not None
    assert stored_item.value["theme"] == "dark"

    # 6. Verify Cascade Deletion of project removes thread, messages, memories, workflow runs
    await prisma.project.delete(where={"id": project.id})

    # Thread, messages, and memories should be cascaded
    deleted_thread = await prisma.conversationthread.find_unique(where={"id": thread.id})
    assert deleted_thread is None, "Thread should be cascade-deleted when project is deleted"

    deleted_msgs = await prisma.chatmessage.find_many(where={"threadId": thread.id})
    assert len(deleted_msgs) == 0, "Messages should be cascade-deleted"

    deleted_memories = await prisma.memoryrecord.find_many(where={"projectId": project.id})
    assert len(deleted_memories) == 0, "Memories should be cascade-deleted"

    # Cleanup user
    await prisma.user.delete(where={"id": user.id})

    await disconnect_db()

