import sys
import os
import asyncio
import grpc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'proto')))

from proto.pb.svc.unit.chat import chat_pb2_grpc, chat_pb2
from proto.pb.svc.unit.common import msg_pb2
from llm_queries import summary_llm, query_funding_view
from auth.oauth import validate_token

class ChatService(chat_pb2_grpc.ChatServiceServicer):
    async def StreamMessage(self, request, context):
        try:
            # 요청 메타데이터와 클라이언트 정보 기록
            print(f"Request received from: {context.peer()}")
            print(f"Metadata: {dict(context.invocation_metadata())}")

            # 요청 데이터 출력
            print(f"Received gRPC request object: {request}")
            
            # 필수 필드 유효성 검사
            if not request.message or not request.sender:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Missing required fields: 'message' or 'sender'")
                print("Invalid request: Missing required fields.")
                return
            
            token = request.metadata['authorization']  # gRPC 메타데이터에서 토큰 추출
            auth_user = validate_token(token)  # validate_token 호출
            user_id = auth_user.userId

            sender = request.sender
            message = request.message
            print(f"Extracted request details - message: {message}, sender: {sender}")

            # 스트리밍 응답 생성
            async for chunk in query_funding_view(message, user_id):
                if chunk.strip():
                    yield chat_pb2.ChatResponse(
                        reply=chunk,
                        status=msg_pb2.Status(code=200, message="Chunk received")
                    )

            # 스트리밍 완료
            print("Streaming completed successfully.")
            yield chat_pb2.ChatResponse(
                reply="Streaming complete.",
                status=msg_pb2.Status(code=200, message="Streaming finished")
            )
        except Exception as e:
            # 예외 처리 및 로그 기록
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in StreamMessage: {error_details}")

            # gRPC 오류 설정
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
