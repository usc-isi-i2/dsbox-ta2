from dsbox_dev_setup import path_setup
path_setup()

from concurrent import futures
import time
import grpc

from dsbox.server.controller.pipeline_compute import PipelineCompute
from dsbox.server.controller.pipeline_data import PipelineData
from dsbox.server.controller.dataflow import Dataflow

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  PipelineCompute().add_to_server(server)
  PipelineData().add_to_server(server)
  Dataflow().add_to_server(server)

  server.add_insecure_port('[::]:50051')
  server.start()

  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  serve()
