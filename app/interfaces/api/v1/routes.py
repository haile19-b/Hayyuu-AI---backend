from fastapi import APIRouter
from app.domain.entities.auth.auth_route import router as auth_router
from app.domain.entities.projects.projects_route import router as projects_router
from app.domain.entities.documents.documents_route import router as documents_router

route = APIRouter()

route.include_router(auth_router)
route.include_router(projects_router)
route.include_router(documents_router)