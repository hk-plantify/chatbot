import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

load_dotenv()
openai_key = os.environ.get("OPENAI_API_KEY")

# 실제 DB랑 연결
funding_csv_file_path = ".tmp/funding_data.csv"
funding_df = pd.read_csv(funding_csv_file_path)

# Hugging Face 임베딩 모델 로드
def load_huggingface_embedding_model(model_name="BAAI/bge-m3"):
    return HuggingFaceEmbeddings(model_name=model_name)

embeddings = load_huggingface_embedding_model()

def format_funding_info(row):
    title = f"펀딩 제목: {row['title']}"
    content = f"펀딩 내용: {row['content']}"
    metadata = {
        "funding_id": row['funding_id'],
        "cur_amount": row['cur_amount'],
        "target_amount": row['target_amount'],
        "percent": row['percent'],
        "status": row['status'],
        "category": row['category'],
        "funding_start_date": row['funding_start_date'],
        "funding_end_date": row['funding_end_date']
    }
    return title, content, metadata

# 텍스트와 메타데이터 분리
titles = []
contents = []
metadata_list = []

for _, row in funding_df.iterrows():
    title, content, metadata = format_funding_info(row)
    titles.append(title)
    contents.append(content)
    metadata_list.append(metadata)

# 벡터 스토어 생성
persist_directory = "./funding_vector_db"
vector_store = Chroma.from_texts(
    texts=titles + contents,  # 텍스트 리스트
    embedding=embeddings,  # 올바르게 전달
    metadatas=metadata_list * 2,  # 메타데이터 리스트
    persist_directory=persist_directory  # 저장 경로
)

vector_store.persist()

print("펀딩 벡터 DB가 생성되고 저장되었습니다.")