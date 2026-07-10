from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from app.domain.entities.auth.auth_services import AuthServices
from app.domain.entities.auth.jwt import decode_token, create_access_token, create_refresh_token
from app.domain.entities.auth.auth_schema import ApiResponse, TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login", auto_error=False)

class AuthController:
    @staticmethod
    async def register(body):
        try:
            await AuthServices.register_user(body.email, body.password)
            return ApiResponse(success=True, data="User registered successfully")
        except ValueError as e:
            return ApiResponse(success=False, error=str(e))

    @staticmethod
    async def login(body):
        try:
            user = await AuthServices.authenticate_user(body.email, body.password)
            access = create_access_token(user.id)
            refresh = create_refresh_token(user.id)
            return ApiResponse(
                success=True,
                data=TokenData(access_token=access, refresh_token=refresh)
            )
        except ValueError as e:
            return ApiResponse(success=False, error=str(e))

    @staticmethod
    async def refresh(body):
        try:
            payload = decode_token(body.refresh_token)
            if payload.get("type") != "refresh":
                return ApiResponse(success=False, error="Invalid token type")
            user_id = payload.get("sub")
            access = create_access_token(user_id)
            refresh = create_refresh_token(user_id)
            return ApiResponse(
                success=True,
                data=TokenData(access_token=access, refresh_token=refresh)
            )
        except Exception:
            return ApiResponse(success=False, error="Invalid refresh token")

    @staticmethod
    async def get_current_user(token: str = Depends(oauth2_scheme)):
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            payload = decode_token(token)
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Invalid token type")
            
            user_id = payload.get("sub")
            user = await AuthServices.get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            return user
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
