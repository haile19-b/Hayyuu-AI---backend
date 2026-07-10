from app.domain.entities.auth.auth_services import AuthServices
from app.domain.entities.auth.jwt import decode_token, create_access_token, create_refresh_token
from app.domain.entities.auth.auth_schema import ApiResponse, TokenData

class AuthController:
    @staticmethod
    async def register(body):
        try:
            await AuthServices.register_user(
                email=body.email,
                password_raw=body.password,
                full_name=body.FullName,
                user_name=body.UserName
            )
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
    async def me(credentials: dict):
        try:
            user_id = credentials.get("sub")
            if not user_id:
                return ApiResponse(success=False, error="Unauthorized")
            
            result = await AuthServices.get_profile(user_id)
            if not result["success"]:
                return ApiResponse(success=False, error=result["error"])
                
            return ApiResponse(success=True, data=result["response"])
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
