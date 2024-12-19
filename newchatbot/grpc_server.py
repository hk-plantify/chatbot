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
    ]
)

logger = logging.getLogger(__name__)  # 로거 생성


class ChatService(chat_pb2_grpc.ChatServiceServicer):
    async def StreamMessage(self, request, context):
        try:
            # 사용자 요청에 대한 응답 처리
            async for chunk_content in query_funding_view(request.message):
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
