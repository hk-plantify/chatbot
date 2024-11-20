import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()
openai_key = os.environ.get("OPENAI_API_KEY")

csv_file_path = ".tmp/preprocessed_card_data.csv"
df = pd.read_csv(csv_file_path)

def load_huggingface_embedding_model(model_name="BAAI/bge-m3"):
    return HuggingFaceEmbeddings(model_name=model_name)

embeddings = load_huggingface_embedding_model()

def format_card_info(row):
    card_core = f"카드 이름: {row['name']}\n회사 이름: {row['company_name']}\n카드 종류: {row['card_type']}\n"
    benefit_info = f"혜택 카테고리: {row['benefit_category']}\n혜택 설명: {row['benefit_description']}\n"
    additional_info = f"추가 정보: {row['additional_info']}\n"
    return card_core, benefit_info, additional_info

card_core_chunks = df.apply(lambda row: format_card_info(row)[0], axis=1).tolist()
benefit_chunks = df.apply(lambda row: format_card_info(row)[1], axis=1).tolist()
additional_chunks = df.apply(lambda row: format_card_info(row)[2], axis=1).tolist()

persist_directory_core = "./card_core_vector_db"
persist_directory_benefit = "./benefit_vector_db"
persist_directory_additional = "./additional_info_vector_db"

vector_store_core = Chroma.from_texts(card_core_chunks, embeddings, persist_directory=persist_directory_core)
vector_store_core.persist()

vector_store_benefit = Chroma.from_texts(benefit_chunks, embeddings, persist_directory=persist_directory_benefit)
vector_store_benefit.persist()

vector_store_additional = Chroma.from_texts(additional_chunks, embeddings, persist_directory=persist_directory_additional)
vector_store_additional.persist()

print("Hugging Face 임베딩 모델을 사용하여 벡터 스토어가 의미 단위별로 생성되고 저장되었습니다.")
