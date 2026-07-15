from fastapi import Request, HTTPException, status
from app.domain.entities.auth.jwt import verify_token

async def auth_middleware(request: Request):
    # 1. Try to get token from Authorization: Bearer <token> header
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]

    # 2. If not in header, try to get from cookies (match accessToken=...)
    if not token:
        token = request.cookies.get("accessToken")

    # 3. If no token found, raise 401 Unauthorized
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token"
        )

    try:
        # Decode and verify JWT
        decoded = verify_token(token)
        return decoded  # This returns the payload (e.g. {"id": userId})
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )