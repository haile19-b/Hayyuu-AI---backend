from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

class RegisterRequest(BaseModel):
    FullName: str = Field(
        ...,
        min_length=3,
        max_length=50
    )
    UserName: str = Field(
        default=None,
        min_length=3,
        max_length=50
    )
    email: EmailStr
    password: str

    @field_validator("FullName")
    @classmethod
    def validate_father_name(cls, v: str) -> str:
        if len(v.strip().split()) < 2:
            raise ValueError("Father Name missed")
        return v

    @model_validator(mode="after")
    def set_default_username(self) -> "RegisterRequest":
        if not self.UserName and self.FullName:
            self.UserName = self.FullName.strip().split()[0]
        return self

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
    username: str
    email: str
    fullName: str
    billingPlan: str
    storageUsed: str
    storageQuota: str
