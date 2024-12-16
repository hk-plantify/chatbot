from pydantic import BaseModel

class AuthUserResponse(BaseModel):
    userId: int
    role: str