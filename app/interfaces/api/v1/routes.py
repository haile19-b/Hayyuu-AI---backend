from fastapi import APIRouter
from app.domain.entities.auth.auth_route import router as auth_router

route = APIRouter()

route.include_router(auth_router)