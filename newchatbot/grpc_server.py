import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'proto')))

from concurrent import futures
import grpc
from proto.pb.svc.unit.chat import chat_pb2_grpc, chat_pb2
from proto.pb.svc.unit.common import msg_pb2
from llm_queries import summary_llm, query_funding_view
from langchain.schema import HumanMessage, SystemMessage

class ChatService(chat_pb2_grpc.ChatServiceServicer):
    async def StreamMessage(self, request, context):
        try:
            user_question = request.message
            async for chunk in query_funding_view(user_question):
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

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatService(), server)
    server.add_insecure_port("[::]:50052")  # Bind to all available IP addresses on port 50052
    print("gRPC server is running on port 50052...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
