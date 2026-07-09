import logging
from typing import AsyncGenerator
from prisma import Prisma
from prisma.errors import PrismaError

# Configure logging
logger = logging.getLogger("uvicorn.error")

# Global prisma client instance
prisma = Prisma()

async def connect_db() -> None:
    """Call this on FastAPI startup."""
    if not prisma.is_connected():
        try:
            await prisma.connect()
            logger.info("✅ Database Connected Successfully")
        except PrismaError as e:
            logger.error(f"❌ Database Connection Failed: {e}")
            raise e
        except Exception as e:
            logger.error(f" An unexpected error occurred while connecting: {e}")
            raise e

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
