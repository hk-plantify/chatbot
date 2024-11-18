import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()
openai_key = os.environ.get("OPENAI_API_KEY")

funding_csv_file_path = ".tmp/funding_data.csv"
funding_df = pd.read_csv(funding_csv_file_path)

funding_df_reduced = funding_df[['title', 'content', 'cur_amount', 'target_amount', 'percent', 'status',
                                 'category', 'funding_start_date', 'funding_end_date', 'donation_start_date', 'donation_end_date']]

embeddings = OpenAIEmbeddings()

def format_funding_info(row):
    funding_info = (
        f"기부 프로젝트 제목: {row['title']}\n"
        f"프로젝트 설명: {row['content']}\n"
        f"현재 모금액: {row['cur_amount']}원\n"
        f"목표 모금액: {row['target_amount']}원\n"
        f"모금 진행률: {row['percent']}%\n"
        f"모금 상태: {row['status']}\n"
        f"카테고리: {row['category']}\n"
        f"모금 시작일: {row['funding_start_date']}\n"
        f"모금 종료일: {row['funding_end_date']}\n"
        f"기부 시작일: {row['donation_start_date']}\n"
        f"기부 종료일: {row['donation_end_date']}\n"
    )
    return funding_info

all_funding_chunks = funding_df_reduced.apply(format_funding_info, axis=1).tolist()

funding_persist_directory = "./funding_vector_db"
funding_vector_store = Chroma.from_texts(all_funding_chunks, embeddings, persist_directory=funding_persist_directory)
funding_vector_store.persist()
print("벡터 스토어가 기부 데이터 기반으로 생성되어 저장되었습니다.")
