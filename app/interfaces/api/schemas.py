from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T")  # Declares the generic type parameter

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None