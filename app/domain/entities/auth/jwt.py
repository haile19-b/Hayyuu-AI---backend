from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from jose import jwt
from pwdlib import PasswordHash
from app.core.env import settings

# Initialize password hash context using recommended settings
password_hash = PasswordHash.recommended()
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    return password_hash.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return password_hash.verify(password, hashed)

def sign_access_token(payload: dict) -> str:
    to_encode = payload.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=ALGORITHM)

def sign_refresh_token(payload: dict) -> str:
    to_encode = payload.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=ALGORITHM)

def verify_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
