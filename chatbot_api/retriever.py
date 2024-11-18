from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.environ.get("OPENAI_API_KEY")
card_vector_db = os.environ.get("CARD_PERSIST_DIRECTORY")
funding_vector_db = os.environ.get("FUNDING_PERSIST_DIRECTORY")

embeddings = OpenAIEmbeddings()

card_vector_store = Chroma(persist_directory=card_vector_db, embedding_function=embeddings)
funding_vector_store = Chroma(persist_directory=funding_vector_db, embedding_function=embeddings)

template = """당신은 카드 정보 및 기부 프로젝트 정보에 대해 간단하고 명료하게 설명해주는 챗봇입니다.
주어진 검색 결과에 있는 정보만을 사용해 응답하세요. 정보가 부족하면 '해당 정보는 없습니다'라고 정중하게 말하세요.
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(template)

# OpenAI 말고 다른 LLM으로 교체 가능하면 교체
llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    streaming=True,
    temperature=0.3,
    callbacks=[StreamingStdOutCallbackHandler()]
)

card_retriever = card_vector_store.as_retriever(search_kwargs={"k": 5})
funding_retriever = funding_vector_store.as_retriever(search_kwargs={"k": 5})

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
