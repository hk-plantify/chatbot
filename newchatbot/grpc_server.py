import sys
import os
import asyncio
import grpc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'proto')))

from proto.pb.svc.unit.chat import chat_pb2_grpc, chat_pb2
from proto.pb.svc.unit.common import msg_pb2
from llm_queries import summary_llm, query_funding_view

class ChatService(chat_pb2_grpc.ChatServiceServicer):
    async def StreamMessage(self, request, context):
        try:
            message = request.message
            sender = request.sender if request.sender else "Unknown"
            print(f"Received request: message={message}, sender={sender}")
            async for chunk in query_funding_view(message, sender):
                if chunk.strip():  # 빈 청크 제거
                    yield chat_pb2.ChatResponse(
                        reply=chunk,
                        status=msg_pb2.Status(code=200, message="Chunk received")
                    )
            # 스트리밍 완료 메시지
            yield chat_pb2.ChatResponse(
                reply="Streaming complete.",
                status=msg_pb2.Status(code=200, message="Streaming finished")
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in StreamMessage: {error_details}")
            yield chat_pb2.ChatResponse(
                reply="An error occurred during streaming.",
                status=msg_pb2.Status(code=500, message=str(e))
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
