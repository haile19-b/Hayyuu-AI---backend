from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from jose import jwt, JWTError
from pwdlib import PasswordHash
from app.core.env import settings

# Initialize password hash context using recommended settings
password_hash = PasswordHash.recommended()
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    return password_hash.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return password_hash.verify(password, hashed)

def create_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=ALGORITHM)

def create_access_token(user_id: str) -> str:
    return create_token({"sub": user_id, "type": "access"}, timedelta(minutes=15))

def create_refresh_token(user_id: str) -> str:
    return create_token({"sub": user_id, "type": "refresh"}, timedelta(days=7))

def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise ValueError("Invalid token")
