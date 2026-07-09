from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.env import settings
from app.core.database import connect_db, disconnect_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Connect to DB
    await connect_db()
    yield
    # 2. Shutdown: Disconnect from DB
    await disconnect_db()

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
# app.include_router(router, prefix="/api")


@app.get("/")
async def health_check():
    return {
        "success": True,
        "message": "API is running 🚀",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)