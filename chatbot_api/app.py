import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.status import HTTP_200_OK
from retriever import qa_chain_card, qa_chain_funding
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


# Pydantic 모델 정의
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
            response = qa_chain_card.invoke({"query": query})
        elif any(keyword in query.lower() for keyword in ["기부", "프로젝트", "모금", "펀딩"]):
            response = qa_chain_funding.invoke({"query": query})
        else:
            raise HTTPException(status_code=400, detail="카드 또는 기부 관련 질문을 해주세요.")

        answer = response['result']
        return {
            "answer": answer,
        }
        # return {
        #     "question": query,
        #     "answer": answer,
        #     "source_documents": response['source_documents']
        # }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="172.30.81.20", port=8000)