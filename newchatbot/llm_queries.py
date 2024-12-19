import os
from db.database import engine
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from sqlalchemy import text
from langchain.memory import ConversationBufferWindowMemory

import time
from openai import RateLimitError


import logging

# logging 설정
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨부터 로그 출력
    format="%(asctime)s [%(levelname)s] %(filename)s: %(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 콘솔에 로그 출력
        logging.FileHandler("app_debug.log")  # 파일에 로그 저장
    ]
)

logger = logging.getLogger(__name__)  # 로거 생성

sql_llm = ChatOpenAI(
    temperature=0,
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

# 대화 버퍼 윈도우 메모리 초기화
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

def extract_sql_from_response(response: str) -> str:
    response = response.replace("```sql", "").replace("```", "")
    return response.strip()

def question_to_sql(user_question: str, user_id: int = None) -> str:
    """
    SQL 쿼리를 생성하는 단계에서는 대화 기록을 사용하지 않습니다.
    """
    # user_id_condition = f"user_id = {user_id}" if user_id else "전체 데이터를 대상으로"
    messages = memory.load_memory_variables({})['history']
    messages = messages + [
        SystemMessage(content="당신은 데이터베이스 전문가이며 MySQL 쿼리 생성기로 동작합니다. \
                      아래 데이터베이스 스키마와 데이터를 참고하여 사용자 질문에 적합한 SELECT SQL 쿼리를 반환하세요. \
                      반환 형식은 반드시 SELECT SQL 쿼리 형식이어야 합니다."),
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
            myFunding_id: BIGINT, -- 사용자 기부 ID (NULL 가능)
            user_id: BIGINT, -- 사용자 ID (NULL 가능)
            price: BIGINT, -- 사용자의 기부 금액
            myFunding_status: ENUM('INPROGRESS', 'COMPLETED', 'DELIVERING', 'DELIVERED', NULL 가능) -- 사용자 기부 상태
        )

        사용자 질문: "{user_question}"

        반환 형식:
        SELECT column1, column2, ...
        FROM funding_view
        WHERE condition
        ORDER BY column ASC/DESC
        LIMIT n;

        작업 요구 사항:
        1. 대화 기록에서 중요한 컨텍스트를 활용하여 사용자 요청에 맞는 SQL 쿼리를 생성하세요.
        2. 만약 현재 질문이 이전 질문을 반복하는 경우, 이전 질문에 대한 답변을 다시 제공하세요.
        3. 질문이 개인화된 경우, 적절한 조건(`user_id = {user_id}`)을 포함하세요.
        4. 질문이 개인화되지 않은 경우, user_id 조건을 제외하고 일반적인 조건으로 쿼리를 작성하세요.
        """)
    ]

    response = sql_llm(messages)
    sql_response = extract_sql_from_response(response.content)

    print(f"Generated SQL Query: {sql_response}")

    if not sql_response.lower().startswith("select"):
        raise ValueError("유효한 SELECT SQL 쿼리가 반환되지 않았습니다.")

    # 메모리에 새로운 사용자 질문과 답변 저장
    memory.save_context({"input": user_question}, {"output": sql_response})

    return sql_response

async def query_funding_view(user_question: str, user_id: int = None, max_retries: int = 3, retry_delay: int = 5):
    logger.info(f"Received user_question: {user_question}, user_id: {user_id}")
    
    for attempt in range(max_retries):
        try:
            # SQL 쿼리 생성
            query_sql = question_to_sql(user_question, user_id)
            logger.debug(f"Generated SQL: {query_sql}")

            # 데이터베이스에서 쿼리 실행
            with engine.connect() as connection:
                result = connection.execute(text(query_sql))
                rows = result.fetchall()
                columns = result.keys()
                logger.debug(f"Fetched rows: {len(rows)}")

            # 데이터 정리
            data = [dict(zip(columns, row)) for row in rows]
            logger.debug(f"Data to summarize: {data}")

            # 응답 요약 처리
            prompt = (
                f"사용자 질문: '{user_question}'\n"
                f"데이터: {data}\n"
                f"위 데이터를 참고하여 사용자 질문에 대한 답변을 간결하게 요약된 문장 형태로 작성하세요.\n"
                f"응답:"
            )
            logger.debug("Sent <SOS> token")
            yield "<SOS>"

            # 대화 메모리를 반영한 스트리밍
            messages = memory.load_memory_variables({})['history']
            messages.append(HumanMessage(content=prompt))

            async for chunk in summary_llm.astream(input=messages):
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                logger.debug(f"Streaming chunk: {chunk_content}")
                yield chunk_content

            logger.debug("Sent <EOS> token")
            yield "<EOS>"
            return  # 성공적으로 처리되면 함수 종료

        except RateLimitError as e:
            # RateLimitError 처리: 재시도 또는 초기화
            logger.warning(f"RateLimitError 발생. 재시도 {attempt + 1}/{max_retries}. 대기 {retry_delay}초.")
            time.sleep(retry_delay)  # 재시도 전에 대기
        except Exception as e:
            # 기타 오류 처리
            logger.error(f"Error during query or streaming: {e}", exc_info=True)
            memory.clear()  # 메모리 초기화
            logger.info("오류로 인해 대화 메모리를 초기화했습니다.")
            raise

    # 최대 재시도 후 실패 처리
    memory.clear()
    logger.info("재시도 실패로 인해 대화 메모리를 초기화했습니다.")
    yield "요청이 너무 많아 처리할 수 없습니다. 대화가 초기화되었습니다. 다시 시도해주세요."