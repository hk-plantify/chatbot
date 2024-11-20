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
당신은 지능적인 챗봇으로, 카드 정보나 기부 프로젝트에 대한 질문이 있을 경우 검색 결과를 활용해 답변을 제공합니다.
처음에 먼저 "안녕하세요 무엇을 도와드릴까요?"라고 물어보세요.

### 지시사항
당신은 사용자로부터 검색어를 입력 받아 카드 정보 및 기부 프로젝트 정보에 대해 간단하고 명료하게 설명해주는 챗봇입니다.
카드 정보나 기부 프로젝트는 주어진 검색 결과에 있는 정보만을 사용해서 응답하고, 간단하게 3줄 이내로 요약해서 답변하세요.
다른 주제의 질문에는 대답하지 말고, 카드 및 기부 관련 질문에만 대답하세요. 다른 주제의 질문이 들어오면, 카드 및 기부 관련만 질문하세요. 라고 알려주세요.
정보가 부족하면 '해당 정보는 없습니다'라고 정중하게 말하세요.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("ai", "안녕하세요 무엇을 도와드릴까요?"),
    ("human", "{query}")
])

# LLM 설정
llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    streaming=True,
    temperature=0.3,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 벡터 리트리버 설정
card_retriever = card_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
funding_retriever = funding_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

# RetrievalQA 체인 설정
qa_chain_card = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt, "document_variable_name": "query"},
    retriever=card_retriever,
    return_source_documents=True
)

qa_chain_funding = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt, "document_variable_name": "query"},
    retriever=funding_retriever,
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
    """음식 관련 질문에 대한 메뉴 추천을 생성하는 함수"""
    response = menu_llm([HumanMessage(content=f"{menu_prompt}\n{query}")])
    return response.content