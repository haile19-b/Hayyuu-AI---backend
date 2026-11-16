import sys
import asyncio
import logging
from typing import AsyncGenerator, Optional

# Ensure asyncio Windows compatibility for psycopg
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from prisma import Prisma
from prisma.errors import PrismaError
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.core.env import settings

# Configure logging
logger = logging.getLogger("uvicorn.error")

# Global prisma client instance configured with serverless pool parameters and extended timeout
prisma = Prisma(
    datasource={"url": settings.prisma_database_url},
    http={"timeout": 60.0}
)

# Global AsyncConnectionPool and LangGraph components
_pg_pool: Optional[AsyncConnectionPool] = None
_checkpointer: Optional[AsyncPostgresSaver] = None
_memory_store: Optional[AsyncPostgresStore] = None


async def get_pg_pool() -> AsyncConnectionPool:
    """Returns the active psycopg AsyncConnectionPool, initializing it if necessary."""
    global _pg_pool
    if _pg_pool is None:
        dsn = settings.clean_postgres_dsn
        _pg_pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=3,
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 15,
                "keepalives_interval": 5,
                "keepalives_count": 3,
            },
            check=AsyncConnectionPool.check_connection,
            max_idle=60.0,
            max_lifetime=600.0,
            open=False,
        )
        await _pg_pool.open()
        logger.info("✅ Global AsyncConnectionPool initialized and opened")
    return _pg_pool


async def get_checkpointer() -> AsyncPostgresSaver:
    """Returns the singleton AsyncPostgresSaver checkpointer for LangGraph agent persistence."""
    global _checkpointer
    if _checkpointer is None:
        pool = await get_pg_pool()
        _checkpointer = AsyncPostgresSaver(pool)
    return _checkpointer


async def get_memory_store() -> AsyncPostgresStore:
    """Returns the singleton AsyncPostgresStore for LangGraph long-term episodic memory."""
    global _memory_store
    if _memory_store is None:
        pool = await get_pg_pool()
        _memory_store = AsyncPostgresStore(pool)
    return _memory_store


async def connect_db(max_retries: int = 3, delay_seconds: float = 1.5) -> None:
    """Connect to database with retry logic and bounded timeout for serverless Neon cold-starts."""
    if not prisma.is_connected():
        for attempt in range(1, max_retries + 1):
            try:
                await prisma.connect(timeout=20)
                logger.info("✅ Database Connected Successfully (Prisma)")
                break
            except (PrismaError, Exception) as e:
                try:
                    if prisma.is_connected():
                        await prisma.disconnect()
                except Exception:
                    pass
                if attempt == max_retries:
                    logger.error(f"❌ Database Connection Failed after {max_retries} attempts: {e}")
                    raise e
                logger.warning(
                    f"⚠️ Database connection attempt {attempt}/{max_retries} failed ({e}). "
                    f"Waking up cloud database... retrying in {delay_seconds}s"
                )
                await asyncio.sleep(delay_seconds)

    # Initialize psycopg pool for LangGraph Saver and Store
    try:
        await get_pg_pool()
        await get_checkpointer()
        await get_memory_store()
        logger.info("✅ LangGraph Checkpointer & Store initialized with shared connection pool")
    except Exception as e:
        logger.warning(f"⚠️ LangGraph connection pool initialization deferred or failed: {e}")


async def disconnect_db() -> None:
    """Call this on FastAPI shutdown to cleanly disconnect Prisma and the psycopg pool."""
    global _pg_pool, _checkpointer, _memory_store
    if _pg_pool is not None:
        try:
            await _pg_pool.close(timeout=2.0)
            logger.info("🛑 LangGraph Connection Pool Closed Successfully")
        except Exception as e:
            logger.error(f"❌ Error during LangGraph pool closure: {e}")
        finally:
            _pg_pool = None
            _checkpointer = None
            _memory_store = None

    if prisma.is_connected():
        try:
            await prisma.disconnect()
            logger.info("🛑 Database Disconnected Successfully (Prisma)")
        except Exception as e:
            logger.error(f"❌ Error during database disconnection: {e}")


async def ensure_db_connected() -> None:
    """Ensures database connection is active, reconnecting if disconnected or stale."""
    try:
        if not prisma.is_connected():
            await connect_db()
        else:
            # Lightweight ping to verify socket health
            await prisma.query_raw("SELECT 1")
    except Exception as e:
        logger.warning(f"Database connection health check failed ({e}). Reconnecting...")
        try:
            if prisma.is_connected():
                await prisma.disconnect()
        except Exception:
            pass
        await asyncio.sleep(0.1)
        await connect_db()

    # Also verify the psycopg pool for LangGraph is initialized and open
    global _pg_pool, _checkpointer, _memory_store
    if _pg_pool is None or getattr(_pg_pool, "_closed", False):
        _pg_pool = None
        _checkpointer = None
        _memory_store = None
        await get_pg_pool()
        await get_checkpointer()
        await get_memory_store()


# FastAPI Dependency Injection
async def get_db() -> AsyncGenerator[Prisma, None]:
    """
    Yields the database client instance.
    No need to open/close connection per request.
    Prisma handles connection pooling internally.
    """
    yield prisma
