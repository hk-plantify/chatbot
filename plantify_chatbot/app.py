import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.status import HTTP_200_OK
from retriever import (
    qa_chain_card_core,
    qa_chain_additional_info,
    qa_chain_benefit,
    qa_chain_funding,
    get_menu_recommendation
)
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
    return {"status": "healthy", "code": HTTP_200_OK}

@app.post("/chat")
async def ask_question(request: ChatbotComponent):
    try:
        query = request.question  # 사용자 입력에서 공백 제거
        print("Received query:", query)

        # 카드 관련 키워드
        card_core_keywords = ["카드 기본", "카드 정보", "카드"]
        card_benefit_keywords = ["혜택", "할인"]
        card_additional_keywords = ["추가 정보", "조건"]

        # 기부 관련 키워드
        funding_keywords = ["기부", "프로젝트", "모금", "펀딩"]

        # 카드 기본 정보 질문 처리
        if any(keyword in query for keyword in card_core_keywords):
            response = qa_chain_card_core({"query": query})
            answer = response['result']
        elif any(keyword in query for keyword in card_benefit_keywords):
            response = qa_chain_benefit(query)
            answer = response['result']
        elif any(keyword in query for keyword in card_additional_keywords):
            response = qa_chain_additional_info(query)
            answer = response['result']
        elif any(keyword in query for keyword in funding_keywords):
            response = qa_chain_funding(query)
            answer = response['result']
        else:
            answer = get_menu_recommendation(query)
        # 응답 반환
        return {"answer": answer}

    except Exception as e:
        print("Error occurred:", e)
        raise HTTPException(status_code=500, detail="오류가 발생했습니다. 다시 시도해주세요.")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
