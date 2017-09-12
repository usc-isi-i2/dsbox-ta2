#!/usr/bin/env python

import os
import sys
import os.path

# Setup Paths
PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENTDIR)

from dsbox_dev_setup import path_setup
path_setup()

import time
import grpc
import numpy
import argparse
from concurrent import futures

from dsbox.server.controller.core import Core
from dsbox.server.controller.data_ext import DataExt
from dsbox.server.controller.dataflow_ext import DataflowExt

numpy.set_printoptions(threshold=numpy.nan)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEFAULT_LIB_DIRECTORY = PARENTDIR + os.sep + "library"

def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--library", dest="library", help="Primitives library directory. [default: %(default)s]", default=DEFAULT_LIB_DIRECTORY)
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    library = args.library

    Core(library).add_to_server(server)
    DataExt().add_to_server(server)
    DataflowExt().add_to_server(server)

    server.add_insecure_port('[::]:50051')
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
