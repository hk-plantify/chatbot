from pydantic import BaseModel
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    status: int
    message: str
    data: Optional[T] = None

    @staticmethod
    def ok(data: Optional[T] = None):
        return ApiResponse(status=200, message="성공", data=data)

    @staticmethod
    def fail(status: int, message: str):
        return ApiResponse(status=status, message=message, data=None)