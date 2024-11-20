import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()
openai_key = os.environ.get("OPENAI_API_KEY")

# 실제 DB랑 연결
csv_file_path = ".tmp/preprocessed_card_data.csv"
df = pd.read_csv(csv_file_path)

embeddings = OpenAIEmbeddings()

def format_card_info(row):
    card_info = f"카드 이름: {row['name']}\n회사 이름: {row['company_name']}\n카드 종류: {row['card_type']}\n"
    card_info += f"혜택 카테고리: {row['benefit_category']}\n혜택 설명: {row['benefit_description']}\n추가 정보: {row['additional_info']}\n"
    return card_info

all_chunks = df.apply(format_card_info, axis=1).tolist()

persist_directory = "./card_vector_db"
vector_store = Chroma.from_texts(all_chunks, embeddings, persist_directory=persist_directory)
vector_store.persist()
print("벡터 스토어가 카드 데이터 기반으로 생성되어 저장되었습니다.")