from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED

class ApplicationException(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)

class AuthErrorCode:
    INVALID_TOKEN = (HTTP_400_BAD_REQUEST, "유효하지 않은 토큰입니다.")
    EXPIRED_TOKEN = (HTTP_401_UNAUTHORIZED, "만료된 토큰입니다.")
    UNSUPPORTED_TOKEN = (HTTP_400_BAD_REQUEST, "지원되지 않는 토큰 형식입니다.")
    TOKEN_CLAIMS_EMPTY = (HTTP_400_BAD_REQUEST, "토큰의 클레임이 비어 있습니다.")
    ACCESS_TOKEN_NULL = (HTTP_400_BAD_REQUEST, "액세스 토큰이 비어 있습니다.")