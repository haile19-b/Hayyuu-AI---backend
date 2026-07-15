from fastapi import APIRouter, Depends, Request, Response, status
from app.domain.entities.auth.auth_schema import (
    ApiResponse, RegisterRequest, LoginRequest, RefreshRequest, 
    AuthResponseData, RefreshResponseData, UserProfile
)
from app.domain.entities.auth.auth_controller import AuthController
from app.middleware.auth_middleware import auth_middleware

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=ApiResponse[AuthResponseData])
async def register(body: RegisterRequest, request: Request, response: Response):
    result = await AuthController.register(body, request)
    if not result.get("success"):
        response.status_code = status.HTTP_400_BAD_REQUEST
    else:
        response.status_code = status.HTTP_201_CREATED
    return result

@router.post("/login", response_model=ApiResponse[AuthResponseData])
async def login(body: LoginRequest, request: Request, response: Response):
    result = await AuthController.login(body, request)
    if not result.get("success"):
        response.status_code = status.HTTP_401_UNAUTHORIZED
    return result

@router.post("/refresh", response_model=ApiResponse[RefreshResponseData])
async def refresh(body: RefreshRequest, response: Response):
    result = await AuthController.refresh(body)
    if not result.get("success"):
        response.status_code = status.HTTP_401_UNAUTHORIZED
    return result

@router.post("/logout", response_model=ApiResponse[None])
async def logout(body: RefreshRequest):
    return await AuthController.logout(body)

@router.post("/logout-all", response_model=ApiResponse[None])
async def logout_all(credentials = Depends(auth_middleware), response: Response = None):
    result = await AuthController.logout_all(credentials)
    if not result.get("success"):
        response.status_code = status.HTTP_401_UNAUTHORIZED
    return result

@router.get("/me", response_model=ApiResponse[UserProfile])
async def get_me(credentials = Depends(auth_middleware), response: Response = None):
    result = await AuthController.me(credentials)
    if not result.get("success"):
        response.status_code = status.HTTP_401_UNAUTHORIZED
    return result