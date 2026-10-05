import asyncio
import logging
from typing import AsyncGenerator
from prisma import Prisma
from prisma.errors import PrismaError
from app.core.env import settings

# Configure logging
logger = logging.getLogger("uvicorn.error")

# Global prisma client instance configured with serverless pool parameters
prisma = Prisma(datasource={"url": settings.prisma_database_url})

async def connect_db(max_retries: int = 4, delay_seconds: float = 3.0) -> None:
    """Connect to database with retry logic and extended timeout for serverless Neon cold-starts."""
    if not prisma.is_connected():
        for attempt in range(1, max_retries + 1):
            try:
                await prisma.connect(timeout=30)
                logger.info("✅ Database Connected Successfully")
                return
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

async def disconnect_db() -> None:
    """Call this on FastAPI shutdown."""
    if prisma.is_connected():
        try:
            await prisma.disconnect()
            logger.info("🛑 Database Disconnected Successfully")
        except Exception as e:
            logger.error(f"❌ Error during database disconnection: {e}")

# FastAPI Dependency Injection
async def get_db() -> AsyncGenerator[Prisma, None]:
    """
    Yields the database client instance.
    No need to open/close connection per request.
    Prisma handles connection pooling internally.
    """
    yield prisma
