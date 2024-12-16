from fastapi import APIRouter, Depends
from auth.oauth import validate_token
from common.response import ApiResponse
from auth.schemas import AuthUserResponse

router = APIRouter(prefix="/v1/auth", tags=["Auth"])

@router.post("/validate-token", response_model=ApiResponse[AuthUserResponse])
def validate_user(user: AuthUserResponse = Depends(validate_token)):
    return ApiResponse.ok(data=user)