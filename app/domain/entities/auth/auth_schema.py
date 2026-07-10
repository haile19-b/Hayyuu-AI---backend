from typing import Generic, TypeVar, Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenData(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str

class UserProfile(BaseModel):
    id: str
    email: str
    createdAt: datetime
    updatedAt: datetime
