import os
from db.database import engine
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from sqlalchemy import text

# ConversationBufferWindowMemory를 사용하여 최근 대화 기록 관리
memory = ConversationBufferWindowMemory(k=5)  # 최근 5개의 대화만 유지

# LLM 설정
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def extract_sql_from_response(response: str) -> str:
    """응답에서 SQL 쿼리 추출."""
    response = response.replace("```sql", "").replace("```", "")
    return response.strip()

def question_to_sql(user_question: str, user_id: int = None) -> str:
    """질문을 SQL 쿼리로 변환."""
    user_id_condition = f"user_id = {user_id}" if user_id else "전체 데이터를 대상으로"
    
    memory.chat_memory.add_message(
        SystemMessage(content="당신은 데이터베이스 전문가이며 MySQL 쿼리 생성기로 동작합니다.")
    )
    
    # 사용자 질문을 포함한 프롬프트 생성
    query_prompt = f"""
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
    1. 반드시 유효한 SELECT SQL 쿼리를 반환하세요.
    2. 지원되지 않는 질문이거나 SQL 쿼리를 작성할 수 없는 경우, "지원되지 않는 요청입니다. 다른 질문을 해주세요."라고 응답하세요.

    반환 형식 예시:
    SELECT column1, column2, ...
    FROM funding_view
    WHERE condition
    ORDER BY column ASC/DESC
    LIMIT n;
    """
    memory.chat_memory.add_message(HumanMessage(content=query_prompt))
    
    # LLM 호출
    response = llm(messages=memory.load_memory_variables({})["history"])
    sql_response = extract_sql_from_response(response.content)
    
    # 에러 처리 대신 LLM이 반환하는 메시지를 직접 확인
    if "지원되지 않는 요청입니다." in sql_response:
        raise ValueError(f"LLM 응답: {sql_response}")
    
    return sql_response

def query_funding_view(user_question: str, user_id: int = None):
    """SQL 쿼리 실행 및 요약 생성."""
    query_sql = question_to_sql(user_question, user_id)
    with engine.connect() as connection:
        result = connection.execute(text(query_sql))
        rows = result.fetchall()
        columns = result.keys()

    # 데이터를 딕셔너리로 변환
    data = [dict(zip(columns, row)) for row in rows]

    # 요약 프롬프트 생성
    summary_prompt = f"""
    사용자 질문: '{user_question}'
    데이터: {data}
    요약:
    """
    memory.chat_memory.add_message(HumanMessage(content=summary_prompt))
    
    # 요약 생성
    response = llm(messages=memory.load_memory_variables({})["history"])
    return response.content