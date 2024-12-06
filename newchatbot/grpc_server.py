import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'proto')))

from concurrent import futures
import grpc
from proto.pb.svc.unit.chat import chat_pb2_grpc, chat_pb2
from proto.pb.svc.unit.common import msg_pb2
from llm_queries import query_funding_view 

class ChatService(chat_pb2_grpc.ChatServiceServicer):
    def SendMessage(self, request, context):
        try:
            response_text = query_funding_view(request.message)
            return chat_pb2.ChatResponse(
                reply=response_text,
                status=msg_pb2.Status(code=200, message="Success")
            )
        except Exception as e:
            return chat_pb2.ChatResponse(
                reply="An error occurred while processing the request.",
                status=msg_pb2.Status(code=500, message=str(e))
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatService(), server)
    server.add_insecure_port("[::]:50052")
    print("gRPC server is running on port 50052...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
