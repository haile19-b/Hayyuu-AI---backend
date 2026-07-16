from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    response: Optional[T] = None
    message: Optional[str] = None
    error: Optional[str] = None

class RegisterRequest(BaseModel):
    fullName: str = Field(..., min_length=3, max_length=50)
    userName: str = Field(default=None, min_length=3, max_length=50)
    email: EmailStr
    password: str

    @model_validator(mode="after")
    def set_default_username(self) -> "RegisterRequest":
        if not self.userName and self.fullName:
            self.userName = self.fullName.strip().split()[0]
        return self

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refreshToken: str

class AuthResponseData(BaseModel):
    username: str
    email: str
    fullName: str
    accessToken: str
    refreshToken: str
    sessionId: str

class RefreshResponseData(BaseModel):
    accessToken: str
    refreshToken: str

class UserProfile(BaseModel):
    username: str
    email: str
    fullName: str
