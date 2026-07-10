from fastapi import APIRouter, Depends
from app.domain.entities.auth.auth_schema import ApiResponse, RegisterRequest, LoginRequest, RefreshRequest, TokenData, UserProfile
from app.domain.entities.auth.auth_controller import AuthController
from app.middleware.auth_middleware import auth_middleware

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=ApiResponse[str])
async def register(body: RegisterRequest):
    return await AuthController.register(body)

@router.post("/login", response_model=ApiResponse[TokenData])
async def login(body: LoginRequest):
    return await AuthController.login(body)

@router.post("/refresh", response_model=ApiResponse[TokenData])
async def refresh(body: RefreshRequest):
    return await AuthController.refresh(body)

@router.get("/me", response_model=ApiResponse[UserProfile])
async def get_me(credentials = Depends(auth_middleware)):
    return await AuthController.me(credentials)
