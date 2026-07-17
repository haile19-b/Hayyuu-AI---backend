from contextlib import asynccontextmanager
import sys
import asyncio

# Fix psycopg Windows compatibility issues with ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.env import settings
from app.core.database import connect_db, disconnect_db
from app.core.queue import connect_redis, disconnect_redis
from app.interfaces.api.v1.routes import route as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Connect to DB and Redis
    await connect_db()
    await connect_redis()
    yield
    # 2. Shutdown: Disconnect from DB and Redis
    await disconnect_db()
    await disconnect_redis()

app = FastAPI(
    title="My API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all application routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def health_check():
    return {
        "success": True,
        "message": "API is running 🚀",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)