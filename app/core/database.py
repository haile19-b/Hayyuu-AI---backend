from prisma import Prisma

# Global prisma client instance
db = Prisma()

async def connect_db():
    """Call this on FastAPI startup"""
    if not db.is_connected():
        await db.connect()

async def disconnect_db():
    """Call this on FastAPI shutdown"""
    if db.is_connected():
        await db.disconnect()

# FastAPI Dependency Injection
async def get_db():
    """
    Yields the database client instance.
    No need to open/close connection per request (Prisma handles connection pooling internally).
    """
    yield db