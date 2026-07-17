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
        redis_pool = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
        logger.info("✅ Redis Pool Connected Successfully")

async def disconnect_redis() -> None:
    """Close the global Redis pool connection."""
    global redis_pool
    if redis_pool is not None:
        await redis_pool.close()
        redis_pool = None
        logger.info("🛑 Redis Pool Disconnected Successfully")

from app.domain.entities.documents.documents_workflow import run_workflow

async def enqueue_document_analysis(document_id: str, project_id: str) -> None:
    """Enqueue a document analysis background job."""
    global redis_pool
    if redis_pool is None:
        await connect_redis()
    await redis_pool.enqueue_job("analyze_document_job", document_id, project_id)
    logger.info(f"Enqueued document analysis job for document {document_id}")

from prisma.enums import DocumentStatus
from app.core.database import prisma
from app.core.progress import publish_progress

async def analyze_document_job(ctx, document_id: str, project_id: str) -> None:
    """The ARQ background task that runs the document analysis pipeline."""
    logger.info(f"Starting background job: analyze_document_job for doc={document_id}, proj={project_id}")
    try:
        await run_workflow(document_id, project_id)
        logger.info(f"Completed background job: analyze_document_job for doc={document_id}")
    except Exception as e:
        logger.error(f"Failed background job: analyze_document_job for doc={document_id}. Error: {e}")
        try:
            await prisma.document.update(
                where={"id": document_id},
                data={"status": DocumentStatus.FAILED}
            )
            await publish_progress(document_id, f"Analysis failed: {str(e)}", "error", "FAILED")
        except Exception as db_err:
            logger.error(f"Failed to update document status to FAILED in DB: {db_err}")
        raise e

async def startup(ctx) -> None:
    """Background worker startup hook to connect to database and Redis."""
    await connect_db()
    await connect_redis()
    logger.info("Worker connections initialized")

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
