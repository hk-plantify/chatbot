from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 가짜 사용자 데이터베이스
fake_users_db = {
    "token123": {"user_id": 101, "username": "testuser"},
    "token456": {"user_id": 102, "username": "demo"}
}

def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_users_db.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user
