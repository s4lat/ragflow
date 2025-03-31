import grpc
from concurrent import futures
import time
import os
import typing as tp


import vector_pb2
import vector_pb2_grpc
from model import Vectorizer


class VectorController(vector_pb2_grpc.VectorServicer):
    def __init__(self):
        self.service = Vectorizer()

    def GetVector(self, request, context):
        data: tp.List[str] = [i for i in request.texts]
        response = self.service(data)
        return vector_pb2.Data(vectors=[vector_pb2.TextVector(data=i) for i in response])


def serve():
    port = os.getenv('PORT', '50055')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
    vector_pb2_grpc.add_VectorServicer_to_server(VectorController(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server is running on port {port}...")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
