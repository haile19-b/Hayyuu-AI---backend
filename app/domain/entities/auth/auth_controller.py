from fastapi import Request
from app.domain.entities.auth.auth_services import AuthServices
from app.domain.entities.auth.auth_schema import ApiResponse

class AuthController:
    @staticmethod
    async def register(body, request: Request):
        try:
            ip = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            
            result = await AuthServices.register(
                email=body.email,
                password_raw=body.password,
                user_name=body.userName,
                full_name=body.fullName,
                ip=ip,
                user_agent=user_agent
            )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def login(body, request: Request):
        try:
            ip = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            
            result = await AuthServices.login(
                email=body.email,
                password_raw=body.password,
                ip=ip,
                user_agent=user_agent
            )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def refresh(body):
        try:
            result = await AuthServices.refresh(body.refreshToken)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def logout(body):
        try:
            result = await AuthServices.logout(body.refreshToken)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def logout_all(credentials: dict):
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
                
            result = await AuthServices.logout_all(user_id)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def me(credentials: dict):
        try:
            user_id = credentials.get("id") if credentials else None
            if not user_id:
                return {"success": False, "error": "Unauthorized"}
                
            result = await AuthServices.get_profile(user_id)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
