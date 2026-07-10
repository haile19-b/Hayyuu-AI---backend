from app.core.database import prisma
from app.domain.entities.auth.jwt import hash_password, verify_password

class AuthServices:
    @staticmethod
    async def register_user(email: str, password_raw: str):
        # Check if email exists
        existing = await prisma.user.find_unique(where={"email": email})
        if existing:
            raise ValueError("Email already registered")
        
        hashed = hash_password(password_raw)
        user = await prisma.user.create(
            data={"email": email, "hashedPassword": hashed}
        )
        return user

    @staticmethod
    async def authenticate_user(email: str, password_raw: str):
        user = await prisma.user.find_unique(where={"email": email})
        if not user:
            raise ValueError("Incorrect email or password")
        
        if not verify_password(password_raw, user.hashedPassword):
            raise ValueError("Incorrect email or password")
            
        return user

    @staticmethod
    async def get_user_by_id(user_id: str):
        return await prisma.user.find_unique(where={"id": user_id})
