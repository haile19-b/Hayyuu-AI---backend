from app.core.database import prisma
from app.domain.entities.auth.jwt import hash_password, verify_password

class AuthServices:
    @staticmethod
    async def register_user(email: str, password_raw: str, full_name: str, user_name: str):
        # Check if email exists
        existing = await prisma.user.find_unique(where={"email": email})
        if existing:
            raise ValueError("Email already registered")
        
        hashed = hash_password(password_raw)
        user = await prisma.user.create(
            data={
                "email": email,
                "hashedPassword": hashed,
                "fullName": full_name,
                "userName": user_name
            }
        )
        return user

    @staticmethod
    async def authenticate_user(email: str, password_raw: str):
        user = await prisma.user.find_unique(where={"email": email})
        if not user or not verify_password(password_raw, user.hashedPassword):
            raise ValueError("Incorrect email or password")
            
        return user

    @staticmethod
    async def get_user_by_id(user_id: str):
        return await prisma.user.find_unique(where={"id": user_id})

    @staticmethod
    async def get_profile(user_id: str):
        user = await prisma.user.find_unique(where={"id": user_id})
        if not user:
            return {"success": False, "error": "User not found"}
        
        # Return exact formatted profile matching Express TS code
        return {
            "success": True,
            "response": {
                "username": user.userName,
                "email": user.email,
                "fullName": user.fullName,
                "billingPlan": "Pro",
                "storageUsed": "0 GB",
                "storageQuota": "5 GB"
            }
        }