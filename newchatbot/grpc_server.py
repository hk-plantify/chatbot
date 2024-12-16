import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'proto')))

from concurrent import futures
import grpc
from proto.pb.svc.unit.chat import chat_pb2_grpc, chat_pb2
from proto.pb.svc.unit.common import msg_pb2
from llm_queries import query_funding_view
from auth.oauth import validate_token 

class ChatService(chat_pb2_grpc.ChatServiceServicer):
    def StreamMessage(self, request, context):
        try:
            # `sender`(userId 토큰) 인증
            user_info = validate_token(request.sender)
            user_id = user_info.userId
            print(f"Authenticated user: {user_id}")

            # 사용자 질문 처리
            user_question = request.message

            # Stream response from `query_funding_view_stream`
            for chunk in query_funding_view(user_question, user_id=user_id):  # user_id 전달
                yield chat_pb2.ChatResponse(
                    reply=chunk,  # 스트리밍 응답 청크
                    status=msg_pb2.Status(code=200, message="Streaming...")
                )
        except Exception as e:
            print(f"Error in StreamMessage: {e}")
            yield chat_pb2.ChatResponse(
                reply="An error occurred while streaming the response.",
                status=msg_pb2.Status(code=500, message=str(e))
            )
            
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatService(), server)
    server.add_insecure_port("[::]:50052")  # Bind to all available IP addresses on port 50052
    print("gRPC server is running on port 50052...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()