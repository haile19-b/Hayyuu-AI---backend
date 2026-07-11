import uuid
import datetime
from jose import JWTError
from app.core.database import prisma
from app.domain.entities.auth.jwt import sign_access_token, sign_refresh_token, verify_token
from app.domain.entities.auth.session_services import SessionService
from app.domain.entities.auth.jwt import hash_password, verify_password

class AuthServices:
    @staticmethod
    async def register(email: str, password_raw: str, user_name: str, full_name: str, ip: str = None, user_agent: str = None):
        existing = await prisma.user.find_first(
            where={
                "OR": [
                    {"email": email},
                    {"userName": user_name}
                ]
            }
        )
        
        if existing:
            if existing.email == email:
                return {"success": False, "error": "User already exists"}
            if existing.userName == user_name:
                return {"success": False, "error": "Username is already taken"}
                
        hashed = hash_password(password_raw)
        user = await prisma.user.create(
            data={
                "email": email,
                "userName": user_name,
                "fullName": full_name,
                "hashedPassword": hashed
            }
        )
        
        session_id = str(uuid.uuid4())
        access_token = sign_access_token({"id": user.id})
        refresh_token = sign_refresh_token({"id": user.id, "sessionId": session_id})
        
        session = await SessionService.create(
            session_id=session_id,
            user_id=user.id,
            refresh_token=refresh_token,
            user_agent=user_agent,
            ip=ip
        )
        
        return {
            "success": True,
            "response": {
                "username": user.userName,
                "email": user.email,
                "fullName": user.fullName,
                "accessToken": access_token,
                "refreshToken": refresh_token,
                "sessionId": session.id
            }
        }

    @staticmethod
    async def login(email: str, password_raw: str, ip: str = None, user_agent: str = None):
        user = await prisma.user.find_unique(where={"email": email})
        if not user or not verify_password(password_raw, user.hashedPassword):
            return {"success": False, "error": "Invalid credentials"}
            
        session_id = str(uuid.uuid4())
        access_token = sign_access_token({"id": user.id})
        refresh_token = sign_refresh_token({"id": user.id, "sessionId": session_id})
        
        session = await SessionService.create(
            session_id=session_id,
            user_id=user.id,
            refresh_token=refresh_token,
            user_agent=user_agent,
            ip=ip
        )
        
        return {
            "success": True,
            "response": {
                "username": user.userName,
                "email": user.email,
                "fullName": user.fullName,
                "accessToken": access_token,
                "refreshToken": refresh_token,
                "sessionId": session.id
            }
        }

    @staticmethod
    async def refresh(refresh_token: str):
        try:
            payload = verify_token(refresh_token)
            session_id = payload.get("sessionId")
            user_id = payload.get("id")
            
            if not session_id or not user_id:
                return {"success": False, "error": "Invalid refresh token"}
                
            session = await prisma.session.find_unique(where={"id": session_id})
            if not session or session.revoked:
                return {"success": False, "error": "Invalid or expired session"}
                
            if session.token != refresh_token:
                await SessionService.revoke(session_id)
                print(f"SECURITY WARNING: Replay attack detected for session {session_id}. Revoking session.")
                return {"success": False, "error": "Compromised session. Please login again."}
                
            now = datetime.datetime.now(datetime.timezone.utc)
            expires_at = session.expiresAt
            if expires_at.tzinfo is None:
                now = datetime.datetime.utcnow()
            if now > expires_at:
                await SessionService.revoke(session_id)
                return {"success": False, "error": "Session expired"}
                
            new_access = sign_access_token({"id": user_id})
            new_refresh = sign_refresh_token({"id": user_id, "sessionId": session_id})
            
            await SessionService.rotate(session_id, new_refresh)
            
            return {
                "success": True,
                "response": {
                    "accessToken": new_access,
                    "refreshToken": new_refresh
                }
            }
        except JWTError:
            return {"success": False, "error": "Invalid refresh token"}

    @staticmethod
    async def logout(refresh_token: str):
        try:
            payload = verify_token(refresh_token)
            session_id = payload.get("sessionId")
            if session_id:
                await SessionService.revoke(session_id)
            return {"success": True, "message": "Logged out successfully"}
        except Exception:
            return {"success": True, "message": "Logged out successfully"}

    @staticmethod
    async def logout_all(user_id: str):
        try:
            await prisma.session.update_many(
                where={"userId": user_id, "revoked": False},
                data={"revoked": True}
            )
            return {"success": True, "message": "Logged out of all devices successfully"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_profile(user_id: str):
        user = await prisma.user.find_unique(where={"id": user_id})
        if not user:
            return {"success": False, "error": "User not found"}
            
        return {
            "success": True,
            "response": {
                "username": user.userName,
                "email": user.email,
                "fullName": user.fullName
            }
        }