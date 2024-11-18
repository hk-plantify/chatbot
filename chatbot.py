import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Depends
from fastapi.security.api_key import APIKey
from starlette.status import HTTP_200_OK

from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

app = FastAPI()

card_persist_directory = "./csv_vector_db"
funding_persist_directory = "./funding_vector_db"

embeddings = OpenAIEmbeddings()

card_vector_store = Chroma(persist_directory=card_persist_directory, embedding_function=embeddings)
funding_vector_store = Chroma(persist_directory=funding_persist_directory, embedding_function=embeddings)

template = """당신은 카드 정보 및 기부 프로젝트 정보에 대해 간단하고 명료하게 설명해주는 챗봇입니다.
주어진 검색 결과에 있는 정보만을 사용해 응답하세요. 정보가 부족하면 '해당 정보는 없습니다'라고 정중하게 말하세요.
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    streaming=True,
    temperature=0.1,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 카드 및 기부 벡터 스토어에서 검색할 수 있도록 설정
card_retriever = card_vector_store.as_retriever(search_kwargs={"k": 5})
funding_retriever = funding_vector_store.as_retriever(search_kwargs={"k": 5})

# 카드 및 기부 데이터를 각각 사용하는 QA 체인
qa_chain_card = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=card_retriever,
    return_source_documents=True
)

qa_chain_funding = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=funding_retriever,
    return_source_documents=True
)

# Pydantic 모델 정의
class ChatbotComponent(BaseModel):
    question: str


@app.get("/health")
def health_check():
    """Health check"""
    return HTTP_200_OK

# API 엔드포인트 생성
@app.post("/chat")
async def ask_question(request: ChatbotComponent):
    try:
        query = request.question

        # 질문 내용에 따라 카드 또는 기부 정보 검색
        if any(keyword in query.lower() for keyword in ["카드", "혜택", "할인", "정보"]):
            response = qa_chain_card({"query": query})
        elif any(keyword in query.lower() for keyword in ["기부", "프로젝트", "모금", "펀딩"]):
            response = qa_chain_funding({"query": query})
        else:
            raise HTTPException(status_code=400, detail="카드 또는 기부 관련 질문을 해주세요.")

        answer = response['result']
        return {
            "question": query,
            "answer": answer,
            "source_documents": response['source_documents']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
