import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.status import HTTP_200_OK
from retriever import qa_chain_card, qa_chain_funding, get_menu_recommendation
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (보안을 위해 프로덕션에서는 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatbotComponent(BaseModel):
    question: str

@app.get("/health")
def health_check():
    """Health check"""
    return HTTP_200_OK


@app.post("/chat")
async def ask_question(request: ChatbotComponent):
    try:
        query = request.question
        print("Received query:", query)

        if any(keyword in query.lower() for keyword in ["카드", "혜택", "할인", "정보"]):
            response = qa_chain_card({"query": query})
            answer = response.get("result", "관련된 정보를 찾을 수 없습니다.")
        elif any(keyword in query.lower() for keyword in ["기부", "프로젝트", "모금", "펀딩"]):
            response = qa_chain_funding({"query": query})
            answer = response.get("result", "관련된 정보를 찾을 수 없습니다.")
        else:
            answer = get_menu_recommendation(query)
            
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)