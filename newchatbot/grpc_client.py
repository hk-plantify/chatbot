import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'proto')))

import grpc
from proto.pb.svc.unit.chat import chat_pb2_grpc, chat_pb2

def run():
    with grpc.insecure_channel("localhost:50052") as channel:
        stub = chat_pb2_grpc.ChatServiceStub(channel)
        response = stub.SendMessage(chat_pb2.ChatRequest(
            message="환경 관련 펀딩 목록",
        ))
        print(f"Server reply: {response.reply}")
        print(f"Status: {response.status.message} (Code: {response.status.code})")

if __name__ == "__main__":
    run()
