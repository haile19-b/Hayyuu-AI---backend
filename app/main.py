from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.env import settings
# from app.routes.index import router

app = FastAPI(
    title="My API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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