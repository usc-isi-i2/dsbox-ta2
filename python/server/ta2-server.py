#!/usr/bin/env python

import argparse
import grpc
import os
import sys
import time

from concurrent import futures

from ta3ta2_api import core_pb2_grpc

# Setup Paths
PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENTDIR)

from dsbox.server.ta2_servicer import TA2Servicer
from dsbox.controller.config import DsboxConfig

# from dsbox_dev_setup import path_setup
# path_setup()


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
PORT = 45042


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port', help='TA2 server port', default=PORT)
    parser.add_argument(
        '--debug-volume-map', action='append',
        help="Map config directories, e.g. --debug-volume-map /host/dir/output:/output --debug-volume-map /host/dir/input:/input",
        default=[])
    parser.add_argument(
        '--load-pipeline', help='Load using fitted pipeline ID')
    args = parser.parse_args()

    print(args)

    server_port = args.port

    dir_mapping = {}
    for entry in args.debug_volume_map:
        host_dir, container_dir = entry.split(':')
        dir_mapping[host_dir] = container_dir
        print('volume: {} to {}'.format(host_dir, container_dir))

    config = DsboxConfig()
    config.load()

    print(config)

    servicer = TA2Servicer(
        config=config,
        directory_mapping=dir_mapping,
        fitted_pipeline_id=args.load_pipeline)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    core_pb2_grpc.add_CoreServicer_to_server(servicer, server)

    server.add_insecure_port('[::]:' + str(server_port))
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
