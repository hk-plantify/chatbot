import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'proto')))

import grpc
from proto.pb.svc.unit.chat import chat_pb2_grpc, chat_pb2

def stream_chat():
    """
    Connects to the gRPC server and receives streaming responses.
    """
    with grpc.insecure_channel("localhost:50052") as channel:
        stub = chat_pb2_grpc.ChatServiceStub(channel)
        request = chat_pb2.ChatRequest(message="안녕하세요, 가장 인기 있는 펀딩은 무엇인가요?", sender="User123")
        
        try:
            for response in stub.StreamMessage(request):
                print(f"Streamed reply: {response.reply}")
        except grpc.RpcError as e:
            print(f"Streaming error: {e.details()}")

if __name__ == "__main__":
    stream_chat()