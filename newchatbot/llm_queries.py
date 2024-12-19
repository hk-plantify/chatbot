import os
from db.database import engine
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from sqlalchemy import text
import logging

# logging 설정
# logging.basicConfig(
#     level=logging.DEBUG,  # DEBUG 레벨부터 로그 출력
#     format="%(asctime)s [%(levelname)s] %(filename)s: %(lineno)d - %(message)s",
#     handlers=[
#         logging.StreamHandler(),  # 콘솔에 로그 출력
#     ]
# )

# logger = logging.getLogger(__name__)  # 로거 생성

sql_llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    streaming=False
)

summary_llm = ChatOpenAI(
    temperature=0.3,  # 좀 더 다양하고 창의적인 응답을 위해 온도를 조정
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def extract_sql_from_response(response: str) -> str:
    response = response.replace("```sql", "").replace("```", "")
    return response.strip()

def question_to_sql(user_question: str) -> str:
    """
    SQL 쿼리를 생성하는 단계에서는 대화 기록을 사용하지 않습니다.
    """
    sql_messages = [
        SystemMessage(content="""
        당신은 데이터베이스 전문가입니다. 사용자의 질문을 MySQL 쿼리로 변환하세요. 
        반환되는 쿼리는 항상 SELECT 문으로 시작해야 하며, 형식은 다음과 같습니다:

        예제 질문: "펀딩 상태가 'COMPLETED'인 데이터를 보여줘."
        예제 출력:
        SELECT *
        FROM funding_view
        WHERE funding_status = 'COMPLETED';
        """),
        HumanMessage(content=f"""
        데이터베이스 스키마:
        - funding_view(
            funding_id: BIGINT, -- 펀딩의 고유 식별자
            title: VARCHAR, -- 펀딩 제목
            content: TEXT, -- 펀딩 설명
            cur_amount: BIGINT, -- 현재 모금된 금액
            target_amount: BIGINT, -- 목표 금액
            percent: FLOAT, -- 달성률
            funding_status: ENUM('INPROGRESS', 'COMPLETED', 'DELIVERING', 'DELIVERED'), -- 펀딩 상태
            category: VARCHAR ('CHILDREN', 'ANIMAL', 'ENVIRONMENT', 'DISABILITY', 'GLOBAL', 'ELDERLY', 'SOCIAL'), -- 펀딩 카테고리
            funding_start_date: DATETIME, -- 펀딩 시작 날짜
            funding_end_date: DATETIME, -- 펀딩 종료 날짜
            donation_start_date: DATETIME, -- 기부 시작 날짜
            donation_end_date: DATETIME, -- 기부 종료 날짜
            organization_id: BIGINT, -- 기부 단체 ID
            organization_name: VARCHAR, -- 기부 단체 이름
            organization_content: TEXT, -- 기부 단체 설명
        )

        사용자 질문: "{user_question}"

        반환 형식:
        SELECT column1, column2, ...
        FROM funding_view
        WHERE condition
        ORDER BY column ASC/DESC
        LIMIT n;

        작업 요구 사항:
        1. 질문에 맞는 SELECT SQL 쿼리를 생성하세요.
        2. 질문의 의도를 분석하여 적절한 조건을 작성하세요.
        """)
    ]

    response = sql_llm.invoke(sql_messages)
    sql_response = extract_sql_from_response(response.content)

    print(f"Generated SQL Query: {sql_response}")

    if not sql_response.lower().startswith("select"):
        raise ValueError("유효한 SELECT SQL 쿼리가 반환되지 않았습니다.")

    return sql_response

async def query_funding_view(user_question: str): 
    try:
        # SQL 생성
        query_sql = question_to_sql(user_question)
        
        # DB에서 쿼리 실행
        with engine.connect() as connection:
            result = connection.execute(text(query_sql))
            rows = result.fetchall()
            columns = result.keys()
    except Exception as e:
        # logger.error(f"Database query error: {e}", exc_info=True)
        raise

    # DB 결과를 dictionary로 변환
    data = [dict(zip(columns, row)) for row in rows]

    # 데이터 요약 처리
    summary_prompt  = (
        f"사용자 질문: '{user_question}'\n"
        f"다음 데이터에서 핵심 정보를 요약하여 간결하게 답변을 작성하세요:\n{data}\n"
        f"응답:"
    )
    
    try:
        # logger.debug("Sent <SOS> token")
        yield "<SOS>"

        summary_messages = [
            HumanMessage(content=summary_prompt)
        ]

        async for chunk in summary_llm.astream(input=summary_messages):
            chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            yield chunk_content
        
        # logger.debug("Sent <EOS> token")
        yield "<EOS>"

    except Exception as e:
        # logger.error(f"Error during OpenAI streaming: {e}", exc_info=True)
        raise
