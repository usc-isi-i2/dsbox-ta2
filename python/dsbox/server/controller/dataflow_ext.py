"""The Python implementation of the GRPC pipeline.DataflowServicer server."""
import grpc

import dataflow_ext_pb2 as dataflow_ext
import dataflow_ext_pb2_grpc as drpc

from dsbox.server.controller.session_handler import Session

class DataflowExt(drpc.DataflowExtServicer):
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
        drpc.add_DataflowExtServicer_to_server(self, server)
