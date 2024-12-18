import os
from db.database import engine
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, FunctionMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from sqlalchemy import text
from langchain.memory import ConversationBufferWindowMemory
import json
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

# Function Calling을 위한 함수 정의
function_definitions = [
    {
        "name": "generate_sql",
        "description": "사용자 요청에 따라 SQL 쿼리를 생성합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "대상 테이블 이름"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "쿼리에 포함될 컬럼 목록"
                },
                "conditions": {
                    "type": "string",
                    "description": "WHERE 조건 (선택 사항)"
                },
                "order_by": {
                    "type": "string",
                    "description": "ORDER BY 조건 (선택 사항)"
                },
                "limit": {
                    "type": "integer",
                    "description": "LIMIT 조건 (선택 사항)"
                }
            },
            "required": ["table_name", "columns"]
        }
    }
]

sql_llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4-0613",
    functions=function_definitions,
    streaming=False
)

summary_llm = ChatOpenAI(
    temperature=0.3,  # 좀 더 다양하고 창의적인 응답을 위해 온도를 조정
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4-0613",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 대화 버퍼 윈도우 메모리 초기화
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

def question_to_sql(user_question: str, user_id: int = None) -> str:
    """
    Function Calling 방식으로 SQL 쿼리를 생성합니다.
    """
    messages = memory.load_memory_variables({})['history']
    messages = messages + [
        SystemMessage(content="당신은 데이터베이스 전문가이며 SQL 쿼리 생성기로 동작합니다."),
        HumanMessage(content=f"사용자 질문: {user_question}")
    ]

    response = sql_llm(messages)

    if response.additional_kwargs.get("function_call"):
        function_call = response.additional_kwargs["function_call"]
        function_name = function_call["name"]
        function_args = json.loads(function_call["arguments"])

        if function_name == "generate_sql":
            table_name = function_args["table_name"]
            columns = function_args["columns"]
            conditions = function_args.get("conditions", "")
            order_by = function_args.get("order_by", "")
            limit = function_args.get("limit")

            # SQL 생성
            sql_query = f"SELECT {', '.join(columns)} FROM {table_name}"
            if conditions:
                sql_query += f" WHERE {conditions}"
            if order_by:
                sql_query += f" ORDER BY {order_by}"
            if limit is not None:
                sql_query += f" LIMIT {limit}"

            logger.debug(f"Generated SQL Query: {sql_query}")
            memory.save_context({"input": user_question}, {"output": sql_query})
            return sql_query

    raise ValueError("Function call failed to generate a SQL query.")

async def query_funding_view(user_question: str, user_id: int = None):
    logger.info(f"Received user_question: {user_question}, user_id: {user_id}")
    try:
        query_sql = question_to_sql(user_question, user_id)
        logger.debug(f"Generated SQL: {query_sql}")

        with engine.connect() as connection:
            result = connection.execute(text(query_sql))
            rows = result.fetchall()
            columns = result.keys()
            logger.debug(f"Fetched rows: {len(rows)}")
    except Exception as e:
        logger.error(f"Database query error: {e}", exc_info=True)
        raise

    data = [dict(zip(columns, row)) for row in rows]
    logger.debug(f"Data to summarize: {data}")

    # 응답 요약 처리
    prompt = f"사용자 질문: '{user_question}'\n데이터: {data}\n응답:"

    try:
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

    except Exception as e:
        logger.error(f"Error during OpenAI streaming: {e}", exc_info=True)
        raise
