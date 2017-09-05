"""The Python implementation of the GRPC pipeline.DataflowServicer server."""
import grpc

import dataflow_service_pb2
import dataflow_service_pb2_grpc

from dsbox.server.controller.session_handler import Session

class Dataflow(dataflow_service_pb2_grpc.DataflowServicer):
    def DescribeDataflow(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDataflowResults(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def add_to_server(self, server):
        dataflow_service_pb2_grpc.add_DataflowServicer_to_server(self, server)
