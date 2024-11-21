from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models.openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import SystemMessage, HumanMessage

import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.environ.get("OPENAI_API_KEY")
additional_info_vector_db = "./additional_info_vector_db"
benefit_vector_db = "./benefit_vector_db"
card_core_vector_db = "./card_core_vector_db"
funding_vector_db = "./funding_vector_db"

# embeddings = OpenAIEmbeddings()

def load_huggingface_embedding_model(model_name="BAAI/bge-m3"):
    return HuggingFaceEmbeddings(model_name=model_name)

embeddings = load_huggingface_embedding_model()

additional_info_store = Chroma(persist_directory=additional_info_vector_db, embedding_function=embeddings)
benefit_store = Chroma(persist_directory=benefit_vector_db, embedding_function=embeddings)
card_core_store = Chroma(persist_directory=card_core_vector_db, embedding_function=embeddings)
funding_store = Chroma(persist_directory=funding_vector_db, embedding_function=embeddings)


system_prompt = """
당신은 카드 및 기부 프로젝트 정보를 제공하는 전문 챗봇입니다. 아래 지침에 따라 사용자의 질문에 간결하고 정확한 답변을 제공합니다.

1. **답변 지침**:
   - 카드 관련 질문 → `card_core_store`, `benefit_store`, `additional_info_store`를 사용해 검색.
   - 기부 관련 질문 → `funding_store`를 사용해 검색.
   - 검색 결과를 바탕으로 간단하고 명확하게 요약하여 답변하세요.
2. **응답 예시**:
   - 질문: "현대카드 중 혜택이 가장 많은 카드는?"
     - 답변: "현대카드 중 혜택이 가장 많은 카드는 현대카드 Z Play입니다. 현대카드 Z Play의 혜택은 총 4개입니다."
   - 질문: "기부 프로젝트 중 가장 인기 있는 것은?"
     - 답변: "가장 인기 있는 기부 프로젝트는 '나무 심기 캠페인'으로, 현재 500명이 참여 중입니다."
3. **에러 핸들링**:
   - 검색 결과가 없을 경우: "관련된 정보를 찾을 수 없습니다. 다른 질문을 해주세요."라고 응답하세요.

당신의 목표는 검색 결과를 기반으로 신뢰할 수 있는 답변을 제공하며, 질문에 적합한 정보를 빠르게 요약하는 것입니다.
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(system_prompt)

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    streaming=True,
    temperature=0.3,  # 안정적인 답변을 위한 낮은 온도 설정
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 각 데이터의 Retriever 설정
additional_info_retriever = additional_info_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
benefit_retriever = benefit_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
card_core_retriever = card_core_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
funding_retriever = funding_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

qa_chain_additional_info = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=additional_info_retriever,
    return_source_documents=True
)

qa_chain_benefit = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=benefit_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

qa_chain_card_core = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=card_core_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

qa_chain_funding = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=funding_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
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
