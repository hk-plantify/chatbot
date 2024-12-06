import os
from db.database import engine
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from sqlalchemy import text

llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def extract_sql_from_response(response: str) -> str:
    response = response.replace("```sql", "").replace("```", "")
    return response.strip()

def question_to_sql(user_question: str, user_id: int = None) -> str:
    user_id_condition = f"user_id = {user_id}" if user_id else "전체 데이터를 대상으로"
    messages = [
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

        데이터 설명:
        - `cur_amount`, `target_amount`: 현재 모금 금액과 목표 금액입니다.
        - `status`: 펀딩 상태 ('INPROGRESS', 'COMPLETED', 'DELIVERING', 'DELIVERED')를 나타냅니다.
        - `category`: 펀딩 카테고리(예: 'CHILDREN', 'ENVIRONMENT')입니다.
        - `funding_start_date`, `funding_end_date`: 펀딩의 시작과 종료 날짜입니다.

        사용자 질문: "{user_question}"

        반환 형식:
        SELECT column1, column2, ...
        FROM funding_view
        WHERE condition
        ORDER BY column ASC/DESC
        LIMIT n;

        작업 요구 사항:
        1. 질문이 개인화된 경우, 적절한 조건(`user_id = {user_id}`)을 포함하세요.
        2. 질문이 개인화되지 않은 경우, user_id 조건을 제외하고 일반적인 조건으로 쿼리를 작성하세요.
        3. 개인화 여부를 판단하여 질문에 가장 적합한 SQL 쿼리를 생성하세요.
        """)
    ]

    response = llm(messages)
    sql_response = extract_sql_from_response(response.content)

    if not sql_response.lower().startswith("select"):
        raise ValueError("유효한 SELECT SQL 쿼리가 반환되지 않았습니다.")
    return sql_response

def query_funding_view(user_question: str, user_id: int = None):
    query_sql = question_to_sql(user_question, user_id)
    with engine.connect() as connection:
        result = connection.execute(text(query_sql))
        rows = result.fetchall()
        columns = result.keys()
    if not rows:
        return "답변할 수 없는 질문입니다."
        
    data = [dict(zip(columns, row)) for row in rows]

    examples = [
    {
        "user_question": "가장 인기 있는 펀딩은 무엇인가요?",
        "data": [
            {"title": "Save the Forest", "cur_amount": 50000, "target_amount": 100000},
            {"title": "Help the Kids", "cur_amount": 30000, "target_amount": 50000}
        ],
        "response": "가장 인기 있는 펀딩은 'Save the Forest'입니다. 현재 50,000원이 모금되었으며 목표 금액은 100,000원입니다."
    },
    {
        "user_question": "펀딩 상태별로 요약해주세요.",
        "data": [
            {"funding_status": "INPROGRESS"},
            {"funding_status": "COMPLETED"},
            {"funding_status": "INPROGRESS"}
        ],
        "response": "현재 펀딩 상태는 다음과 같습니다: INPROGRESS: 2개, COMPLETED: 1개."
    },
    {
        "user_question": "카테고리별 펀딩 현황을 보여주세요.",
        "data": [
            {"category": "CHILDREN"},
            {"category": "ANIMAL"},
            {"category": "CHILDREN"}
        ],
        "response": "카테고리별 현황은 다음과 같습니다: CHILDREN: 2개, ANIMAL: 1개."
    }
    ]
    example_prompts = "\n\n".join(
        f"사용자 질문: '{example['user_question']}'\n"
        f"데이터: {example['data']}\n"
        f"응답: {example['response']}"
        for example in examples
    )
    
    prompt = f"""
    당신은 데이터를 요약하여 사용자 질문에 적합한 답변을 생성하는 AI 어시스턴트입니다.
    응답은 반드시 2~3줄로 요약해서 답변하세요.
    아래는 몇 가지 예제입니다:
    
    {example_prompts}
    
    사용자 질문: '{user_question}'
    데이터: {data}
    응답:
    """
    
    messages = [
        SystemMessage(content="당신은 데이터를 요약하여 간결하고 명확한 답변을 생성하는 AI입니다."),
        HumanMessage(content=prompt)
    ]
    
    response = llm(messages)
    return response.content