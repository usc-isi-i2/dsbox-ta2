from dsbox_dev_setup import path_setup
path_setup()

from concurrent import futures
import time
import grpc

import os.path
import argparse

from dsbox.server.controller.pipeline_compute import PipelineCompute
from dsbox.server.controller.pipeline_data import PipelineData
from dsbox.server.controller.dataflow import Dataflow

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEFAULT_LIB_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/library"

def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--library", dest="library", help="Primitives library directory. [default: %(default)s]", default="library")
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    library = args.library
    if library is None:
        library = DEFAULT_LIB_DIRECTORY

    PipelineCompute(library).add_to_server(server)
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
