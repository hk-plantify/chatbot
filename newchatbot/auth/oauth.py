import os
import httpx
from fastapi import Depends
from fastapi.security import HTTPBearer
from auth.schemas import AuthUserResponse
from common.exception import ApplicationException, AuthErrorCode

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL")
oauth2_scheme = HTTPBearer()

def validate_token(token: str) -> AuthUserResponse:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with httpx.Client() as client:
            response = client.post(f"{AUTH_SERVICE_URL}/v1/auth/validate-token", headers=headers)
            response.raise_for_status()

        data = response.json()
        if data.get("status") != 200 or "data" not in data:
            raise Exception("Invalid token")

        return AuthUserResponse(**data["data"])
    except httpx.HTTPError as e:
        raise Exception(f"Auth service unavailable: {str(e)}")
