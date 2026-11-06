import sys
import asyncio
import logging

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.database import prisma, connect_db, disconnect_db
from app.core.env import settings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_setup")

async def run_setup():
    logger.info("Connecting to Prisma...")
    await connect_db()
    
    # 1. Setup PGVector extension and ChatMessage vector column
    logger.info("Configuring pgvector extension and chat_messages embedding column...")
    try:
        await prisma.execute_raw("CREATE EXTENSION IF NOT EXISTS vector;")
        await prisma.execute_raw("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS embedding vector(768);")
        await prisma.execute_raw(
            "CREATE INDEX IF NOT EXISTS chat_messages_embedding_hnsw_idx ON chat_messages USING hnsw (embedding vector_cosine_ops);"
        )
        logger.info("✅ pgvector extension and chat_messages HNSW index configured successfully.")
    except Exception as e:
        logger.error(f"❌ Error configuring pgvector: {e}")
        raise e

    # 2. Setup LangGraph Checkpointer (checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations)
    dsn = settings.clean_postgres_dsn
    logger.info("Initializing and migrating LangGraph Checkpointer tables...")
    try:
        async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
            await checkpointer.setup()
        logger.info("✅ LangGraph Checkpointer tables setup successfully.")
    except Exception as e:
        logger.error(f"❌ Error setting up LangGraph checkpointer: {e}")
        raise e

    # 3. Setup LangGraph Memory Store (store tables and indexes)
    logger.info("Initializing and migrating LangGraph Store tables...")
    try:
        async with AsyncPostgresStore.from_conn_string(dsn) as store:
            await store.setup()
        logger.info("✅ LangGraph Store tables setup successfully.")
    except Exception as e:
        logger.error(f"❌ Error setting up LangGraph store: {e}")
        raise e

    # 4. Verify table existence
    logger.info("Verifying created tables in public schema...")
    tables = await prisma.query_raw(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"
    )
    table_names = [t["table_name"] for t in tables]
    logger.info(f"Existing tables in public schema ({len(table_names)}): {', '.join(table_names)}")

    await disconnect_db()
    logger.info("🎉 Database setup completed successfully!")

if __name__ == "__main__":
    asyncio.run(run_setup())
