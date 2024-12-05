from fastapi import HTTPException
from llm_queries import query_funding_view
from models.model import QueryRequest

def handle_query(request: QueryRequest, current_user: dict):
    user_id = current_user.get("user_id")
    try:
        result = query_funding_view(request.question, user_id)
        return {"question": request.question, "user": current_user["username"], "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")