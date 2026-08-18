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

    # Start programmatic arq worker inside the FastAPI event loop using async_run
    from arq.worker import create_worker
    from app.core.queue import WorkerSettings
    
    try:
        worker = create_worker(WorkerSettings)
        worker_task = asyncio.create_task(worker.async_run())
        app.state.worker = worker
        app.state.worker_task = worker_task
    except Exception as worker_err:
        import logging
        logger = logging.getLogger("uvicorn.error")
        logger.error(f"⚠️ Could not start arq worker programmatically: {worker_err}. Background worker features disabled.")

    yield
    # 2. Shutdown: Disconnect from DB and Redis
    if hasattr(app.state, "worker"):
        try:
            await app.state.worker.close()
        except Exception:
            pass

    if hasattr(app.state, "worker_task"):
        app.state.worker_task.cancel()
        try:
            await app.state.worker_task
        except asyncio.CancelledError:
            pass

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

app.state.max_form_size = settings.MAX_FORM_SIZE_MB * 1024 * 1024  # Configure form limit from settings

# Register all application routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def health_check():
    return {
        "success": True,
        "message": "API is running 🚀",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True, reload_dirs=["app"])