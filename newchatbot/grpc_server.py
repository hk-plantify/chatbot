import sys
import os
import asyncio
import grpc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'proto')))

from proto.pb.svc.unit.chat import chat_pb2_grpc, chat_pb2
from proto.pb.svc.unit.common import msg_pb2
from llm_queries import summary_llm, query_funding_view
from auth.oauth import validate_token


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


async def StreamMessage(self, request, context):
    try:
        logger.info(f"Request received from: {context.peer()}")

        # 인증 검사
        metadata = dict(context.invocation_metadata())
        token = metadata.get('authorization', None)
        if not token:
            logger.warning("Authorization token is missing. Proceeding without authentication for testing.")
            user_id = None  # 기본값 설정
        else:
            auth_user = validate_token(token)
            user_id = auth_user.userId

        logger.debug(f"Validated user_id: {user_id}")

        # **초기 인사 메시지 전송**
        welcome_message = "안녕하세요! 무엇을 도와드릴까요?"
        logger.debug("Sent welcome message")
        yield chat_pb2.ChatResponse(
            reply=welcome_message,
            status=msg_pb2.Status(code=200, message="Welcome message sent")
        )

        # 사용자 요청 처리
        while True:
            # 클라이언트 입력을 수신 (비동기 스트리밍)
            request = await context.read()  # 새로운 요청을 읽음

            # 스트림 종료 또는 잘못된 request 처리
            if request is None or not hasattr(request, "message"):
                logger.info("Stream closed by client or invalid request received.")
                break  # 스트림 종료

            if not request.message.strip():  # 빈 문자열이면 무시
                logger.debug("Received empty message. Ignoring input.")
                continue

            logger.info(f"Processing message: {request.message}")

            # 사용자 요청에 대한 응답 처리
            async for chunk_content in query_funding_view(request.message, user_id):
                logger.debug(f"Chunk sent: {chunk_content}")
                yield chat_pb2.ChatResponse(
                    reply=chunk_content,  # chunk 자체가 이미 텍스트 데이터
                    status=msg_pb2.Status(code=200, message="Chunk received")
                )

        logger.info("Streaming completed successfully.")

    except Exception as e:
        logger.error("Error in StreamMessage", exc_info=True)
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details("Internal server error during streaming")
        yield chat_pb2.ChatResponse(
            reply="An internal server error occurred.",
            status=msg_pb2.Status(code=500, message="Internal server error")
        )



async def serve():
    # 비동기 gRPC 서버 생성
    server = grpc.aio.server()
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatService(), server)
    server.add_insecure_port("[::]:50052")  # 모든 IP 주소에서 포트 50052로 바인딩
    print("비동기 gRPC 서버가 포트 50052에서 실행 중입니다...")
    await server.start()  # 서버 시작
    await server.wait_for_termination()  # 종료 대기

if __name__ == "__main__":
    asyncio.run(serve())
