import logging
import sys
import asyncio

# Fix psycopg Windows compatibility issues with ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from arq.connections import RedisSettings
from arq import create_pool
from app.core.env import settings
from app.core.database import connect_db, disconnect_db

logger = logging.getLogger("uvicorn.error")

redis_pool = None

async def connect_redis() -> None:
    """Connect to Redis and initialize the global pool."""
    global redis_pool
    if redis_pool is None:
        try:
            # Set a shorter connection timeout for quick fallback
            settings_obj = RedisSettings.from_dsn(settings.REDIS_URL)
            redis_pool = await create_pool(settings_obj)
            logger.info("✅ Redis Pool Connected Successfully")
        except Exception as e:
            logger.error(f"⚠️ Failed to connect to Redis: {e}. Background tasks will be disabled.")
            redis_pool = None

async def disconnect_redis() -> None:
    """Close the global Redis pool connection."""
    global redis_pool
    if redis_pool is not None:
        try:
            await redis_pool.close()
        except Exception:
            pass
        redis_pool = None
        logger.info("🛑 Redis Pool Disconnected Successfully")

async def enqueue_document_analysis(document_id: str, project_id: str) -> None:
    """Enqueue a document analysis background job."""
    global redis_pool
    if redis_pool is None:
        await connect_redis()
    
    if redis_pool is not None:
        try:
            await redis_pool.enqueue_job("analyze_document_job", document_id, project_id)
            logger.info(f"Enqueued document analysis job for document {document_id}")
        except Exception as e:
            logger.error(f"❌ Failed to enqueue document analysis to Redis: {e}")
    else:
        logger.warning(f"⚠️ Redis is offline. Document {document_id} analysis cannot be enqueued.")

from prisma.enums import DocumentStatus
from app.core.database import prisma

async def analyze_document_job(ctx, document_id: str, project_id: str) -> None:
    """The ARQ background task that runs the document analysis pipeline."""
    from app.agents.requirements_extractor.graph import run_workflow
    from app.core.progress import publish_progress
    logger.info(f"Starting background job: analyze_document_job for doc={document_id}, proj={project_id}")
    try:
        await run_workflow(document_id, project_id)
        logger.info(f"Completed background job: analyze_document_job for doc={document_id}")
    except Exception as e:
        logger.error(f"Failed background job: analyze_document_job for doc={document_id}. Error: {e}")
        try:
            from app.core.database import ensure_db_connected
            await ensure_db_connected()
            await prisma.document.update(
                where={"id": document_id},
                data={"status": DocumentStatus.FAILED}
            )
        except Exception as db_err:
            logger.error(f"Failed to update document status to FAILED in DB: {db_err}")
            
        try:
            await publish_progress(document_id, f"Analysis failed: {str(e)}", "error", "FAILED")
        except Exception as pub_err:
            logger.error(f"Failed to publish failure progress: {pub_err}")
        raise e

async def startup(ctx) -> None:
    """Background worker startup hook to connect to database and Redis, and pre-warm agent pipelines."""
    await connect_db()
    await connect_redis()
    logger.info("Worker database & Redis connections initialized")

    try:
        # 1. Initialize LangGraph Postgres checkpointer tables once on worker boot
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(settings.clean_postgres_dsn) as checkpointer:
            await checkpointer.setup()
        logger.info("LangGraph checkpointer tables verified on worker startup")
    except Exception as cp_err:
        logger.warning(f"Failed to pre-setup LangGraph checkpointer tables on worker startup: {cp_err}")

    try:
        # 2. Pre-warm agent pipelines and deep learning modules in memory
        logger.info("Pre-warming agent pipelines and model dependencies...")
        import app.agents.requirements_extractor.graph
        import app.agents.knowledge_builder.graph
        from app.core.text_extractor import get_doc_converter
        get_doc_converter()
        logger.info("Agent pipelines and model dependencies pre-warmed successfully")
    except Exception as warm_err:
        logger.warning(f"Agent pipelines pre-warming encountered an issue (non-fatal): {warm_err}")

async def shutdown(ctx) -> None:
    """Background worker shutdown hook to disconnect database and Redis."""
    await disconnect_db()
    await disconnect_redis()
    logger.info("Worker connections closed")

class WorkerSettings:
    """Settings required by the arq CLI to run the background worker."""
    functions = [analyze_document_job]
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    on_startup = startup
    on_shutdown = shutdown
    job_timeout = 1800  # Give 30 minutes for large document parses and first-time downloads
