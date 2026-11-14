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

async def enqueue_document_analysis(document_id: str, project_id: str, force: bool = False) -> bool:
    """Enqueue a document analysis background job with deduplication and retry support.
    
    If force=True, cancels any previous execution and flushes Redis result locks to allow immediate re-processing.
    Returns True if successfully enqueued, False otherwise.
    """
    global redis_pool
    if redis_pool is None:
        await connect_redis()
    
    if redis_pool is not None:
        try:
            from arq.jobs import Job, JobStatus
            from arq.constants import job_key_prefix, result_key_prefix, in_progress_key_prefix
            job_id = f"doc_analysis_{document_id}"

            job = Job(job_id, redis_pool)
            current_status = await job.status()

            if force or current_status in (JobStatus.complete, JobStatus.not_found):
                # If a previous run completed, failed, or we are forcing a retry:
                # Remove previous ARQ result/job keys so ARQ allows re-enqueuing the same job_id.
                if force and current_status in (JobStatus.queued, JobStatus.in_progress, JobStatus.deferred):
                    try:
                        await job.abort(timeout=2.0)
                    except Exception:
                        pass
                await redis_pool.delete(
                    f"{job_key_prefix}{job_id}",
                    f"{result_key_prefix}{job_id}",
                    f"{in_progress_key_prefix}{job_id}"
                )
                logger.info(f"Cleared previous job state for document {document_id} (previous status: {current_status}). Enqueuing fresh run.")
            elif current_status in (JobStatus.queued, JobStatus.in_progress, JobStatus.deferred):
                logger.info(f"Document analysis job for document {document_id} is already active ({current_status}). Skipping duplicate enqueue.")
                return False

            enqueued_job = await redis_pool.enqueue_job(
                "analyze_document_job",
                document_id,
                project_id,
                _job_id=job_id
            )
            if enqueued_job:
                logger.info(f"✅ Enqueued document analysis job for document {document_id} (job_id={job_id})")
                return True
            else:
                # Safety fallback: force clear lingering key and retry once
                await redis_pool.delete(
                    f"{job_key_prefix}{job_id}",
                    f"{result_key_prefix}{job_id}",
                    f"{in_progress_key_prefix}{job_id}"
                )
                retry_job = await redis_pool.enqueue_job(
                    "analyze_document_job",
                    document_id,
                    project_id,
                    _job_id=job_id
                )
                if retry_job:
                    logger.info(f"✅ Enqueued document analysis job on forced retry for {document_id}")
                    return True
                logger.warning(f"⚠️ Could not enqueue job for document {document_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to enqueue document analysis to Redis: {e}")
            return False
    else:
        logger.warning(f"⚠️ Redis is offline. Document {document_id} analysis cannot be enqueued.")
        return False

async def abort_document_analysis(document_id: str) -> None:
    """Abort and cancel any pending or running ARQ job for a document in Redis."""
    global redis_pool
    if redis_pool is None:
        await connect_redis()
    if redis_pool is not None:
        try:
            from arq.jobs import Job
            job_id = f"doc_analysis_{document_id}"
            job = Job(job_id, redis_pool)
            await job.abort(timeout=5.0)
            logger.info(f"Successfully requested abort for ARQ job {job_id}")
        except Exception as e:
            logger.debug(f"Non-fatal: could not abort ARQ job for document {document_id}: {e}")

from typing import Optional

async def enqueue_memory_consolidation(user_id: str, project_id: Optional[str], thread_id: str) -> None:
    """Enqueue an out-of-band episodic memory consolidation background job."""
    global redis_pool
    if redis_pool is None:
        await connect_redis()
    
    if redis_pool is not None:
        try:
            await redis_pool.enqueue_job("consolidate_memory_job", user_id, project_id, thread_id)
            logger.info(f"Enqueued memory consolidation for thread {thread_id}")
        except Exception as e:
            logger.error(f"❌ Failed to enqueue memory consolidation to Redis: {e}")
    else:
        logger.warning(f"⚠️ Redis is offline. Memory consolidation for thread {thread_id} falling back to asyncio background task.")
        try:
            asyncio.create_task(consolidate_memory_job(None, user_id, project_id or "", thread_id))
        except Exception as bg_err:
            logger.error(f"Failed to launch in-process consolidation fallback: {bg_err}")

from prisma.enums import DocumentStatus
from app.core.database import prisma

async def analyze_document_job(ctx, document_id: str, project_id: str) -> None:
    """The ARQ background task that runs the document analysis pipeline."""
    from app.agents.requirements_extractor.graph import run_workflow
    from app.core.progress import publish_progress
    from app.core.database import prisma, ensure_db_connected

    logger.info(f"Starting background job: analyze_document_job for doc={document_id}, proj={project_id}")

    # Verify document exists before starting. If deleted, discard gracefully.
    try:
        await ensure_db_connected()
        existing_doc = await prisma.document.find_unique(where={"id": document_id})
        if not existing_doc:
            logger.warning(f"Document {document_id} was deleted before analysis began. Discarding job from Redis.")
            return
    except Exception as check_err:
        logger.warning(f"Initial document existence check error: {check_err}")

    try:
        await run_workflow(document_id, project_id)
        logger.info(f"Completed background job: analyze_document_job for doc={document_id}")
    except Exception as e:
        logger.error(f"Failed background job: analyze_document_job for doc={document_id}. Error: {e}")
        try:
            await ensure_db_connected()
            doc_check = await prisma.document.find_unique(where={"id": document_id})
            if not doc_check:
                logger.warning(f"Document {document_id} no longer exists. Skipping failure status update and discarding retry.")
                return

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

async def consolidate_memory_job(ctx, user_id: str, project_id: str, thread_id: str) -> None:
    """The ARQ background task for cold-path episodic memory extraction and consolidation."""
    from app.agents.memory.consolidator import consolidate_thread_memories
    from app.core.database import prisma, ensure_db_connected
    logger.info(f"Starting background job: consolidate_memory_job for thread={thread_id}")
    try:
        await ensure_db_connected()
        messages = await prisma.chatmessage.find_many(
            where={"threadId": thread_id},
            order={"createdAt": "asc"},
            take=30
        )
        msg_dicts = [
            {"role": m.role.value if hasattr(m.role, "value") else str(m.role), "content": m.content}
            for m in messages
        ]
        consolidated = await consolidate_thread_memories(user_id, project_id, thread_id, msg_dicts)
        logger.info(f"Completed background job: consolidate_memory_job. Saved {len(consolidated)} memories.")
    except Exception as e:
        logger.error(f"Failed background job: consolidate_memory_job for thread={thread_id}. Error: {e}")
        raise e

async def startup(ctx) -> None:
    """Background worker startup hook to connect to database and Redis, and pre-warm agent pipelines."""
    # Ensure all worker and agent logs are flushed to stdout in Docker
    root_log = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root_log.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        root_log.addHandler(sh)
        root_log.setLevel(logging.INFO)

    uv_logger = logging.getLogger("uvicorn.error")
    uv_logger.setLevel(logging.INFO)
    if not uv_logger.handlers:
        sh2 = logging.StreamHandler(sys.stdout)
        sh2.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        uv_logger.addHandler(sh2)

    await connect_db()
    await connect_redis()
    logger.info("Worker database & Redis connections initialized")

    try:
        # 1. Initialize LangGraph Postgres checkpointer tables once on worker boot
        from app.core.database import get_checkpointer
        checkpointer = await get_checkpointer()
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
    functions = [analyze_document_job, consolidate_memory_job]
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    on_startup = startup
    on_shutdown = shutdown
    job_timeout = 1800  # Give 30 minutes for large document parses and first-time downloads
