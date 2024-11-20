from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models.openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage

import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.environ.get("OPENAI_API_KEY")
card_vector_db = os.environ.get("CARD_PERSIST_DIRECTORY")
funding_vector_db = os.environ.get("FUNDING_PERSIST_DIRECTORY")

embeddings = OpenAIEmbeddings() 

card_vector_store = Chroma(persist_directory=card_vector_db, embedding_function=embeddings)
funding_vector_store = Chroma(persist_directory=funding_vector_db, embedding_function=embeddings)

system_prompt = """
당신은 카드 및 기부 프로젝트 정보를 제공하는 전문 챗봇입니다. 아래 지침에 따라 사용자의 질문에 간결하고 정확한 답변을 제공합니다.

1. **초기 인사**: 처음 대화를 시작할 때는 항상 "안녕하세요, 무엇을 도와드릴까요?"라고 물어보세요.
2. **답변 지침**:
   - 카드 관련 질문 → `card_vector_store`를 사용해 검색.
   - 기부 관련 질문 → `funding_vector_store`를 사용해 검색.
   - 검색 결과를 바탕으로, 간단하고 명확하게 요약하여 답변하세요.
3. **응답 예시**:
   - 질문: "현대카드 중 혜택이 가장 많은 카드는?"
     - 답변: "현대카드 중 혜택이 가장 많은 카드는 현대카드 Z Play입니다. 현대카드 Z Play의 혜택은 총 4개입니다."
   - 질문: "기부 프로젝트 중 가장 인기 있는 것은?"
     - 답변: "가장 인기 있는 기부 프로젝트는 '나무 심기 캠페인'으로, 현재 500명이 참여 중입니다."
4. **제한사항**:
   - 카드 및 기부와 관련되지 않은 질문이 들어오면, "카드 및 기부 관련 질문만 가능합니다."라고 응답하세요.
5. **에러 핸들링**:
   - 검색 결과가 없을 경우: "관련된 정보를 찾을 수 없습니다. 다른 질문을 해주세요."라고 응답하세요.

당신의 목표는 검색 결과를 기반으로 신뢰할 수 있는 답변을 제공하며, 질문에 적합한 정보를 빠르게 요약하는 것입니다.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "질문: {query}"),
    ("assistant", "검색 결과: {context}")
])

# LLM 설정
llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    streaming=True,
    temperature=0.2,
    callbacks=[StreamingStdOutCallbackHandler()]
)

card_retriever = card_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
funding_retriever = funding_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})


qa_chain_card = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=card_retriever,
    chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
    return_source_documents=False
)

qa_chain_funding = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=funding_retriever,
    chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
    return_source_documents=False
)


menu_prompt = """
당신은 음식 메뉴 추천 전문가입니다. 사용자로부터 점심이나 저녁 메뉴 추천, 음식 관련 질문을 받을 경우에만 응답하세요.
음식 추천은 사용자가 즐길 수 있는 다양한 메뉴를 제공하며, 창의적이고 맛있는 옵션을 제안합니다.
다른 주제의 질문에는 대답하지 말고, 음식과 관련된 질문에만 응답하세요.
음식 관련 질문이 아니면, "카드 및 기부 관련만 질문하세요." 라고 알려주세요.
되묻는 형식의 문장은 답변하지 마세요.
"""

menu_llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    streaming=True,
    temperature=0.7,  # 메뉴 추천에 더 창의적인 응답을 위해 온도를 높임
    callbacks=[StreamingStdOutCallbackHandler()]
)

def get_menu_recommendation(query: str) -> str:
    response = menu_llm([HumanMessage(content=f"{menu_prompt}\n{query}")])
    return response.content