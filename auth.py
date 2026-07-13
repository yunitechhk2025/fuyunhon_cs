import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import database

JWT_SECRET = os.getenv("JWT_SECRET", "insecure_default_secret_please_change")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 12

bearer_scheme = HTTPBearer(auto_error=False)


def create_token(agent_row) -> str:
    payload = {
        "sub": str(agent_row["id"]),
        "username": agent_row["username"],
        "display_name": agent_row["display_name"],
        "role": agent_row["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌无效或已过期") from exc


def get_current_agent(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录或缺少令牌")
    payload = decode_token(credentials.credentials)
    return {
        "id": int(payload["sub"]),
        "username": payload["username"],
        "display_name": payload["display_name"],
        "role": payload["role"],
    }


def require_admin(agent: dict = Depends(get_current_agent)) -> dict:
    if agent.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return agent
