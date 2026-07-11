from datetime import datetime, timedelta, timezone
from app.core.database import prisma

class SessionService:
    @staticmethod
    async def create(session_id: str, user_id: str, refresh_token: str, user_agent: str = None, ip: str = None):
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        return await prisma.session.create(
            data={
                "id": session_id,
                "userId": user_id,
                "token": refresh_token,
                "expiresAt": expires_at,
                "userAgent": user_agent,
                "ip": ip,
                "revoked": False
            }
        )

    @staticmethod
    async def find_valid(refresh_token: str):
        return await prisma.session.find_first(
            where={
                "token": refresh_token,
                "revoked": False,
                "expiresAt": {"gt": datetime.now(timezone.utc)}
            }
        )

    @staticmethod
    async def rotate(session_id: str, new_token: str):
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        return await prisma.session.update(
            where={"id": session_id},
            data={
                "token": new_token,
                "expiresAt": expires_at
            }
        )

    @staticmethod
    async def revoke(session_id: str):
        return await prisma.session.update(
            where={"id": session_id},
            data={"revoked": True}
        )
