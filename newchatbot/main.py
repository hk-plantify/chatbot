from fastapi import FastAPI, Depends
from auth.auth import get_current_user
from models.model import QueryRequest
from services.service import handle_query

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Chatbot API is running!"}

def get_response(message: str) -> str:
    # 여기에 메시지를 처리하는 로직을 구현하세요
    return f"Echo: {message}"

@app.post("/query")
def chatbot_query(request: QueryRequest, current_user: dict = Depends(get_current_user)):
    return handle_query(request, current_user)
