"""The Python implementation of the GRPC pipeline.DataflowServicer server."""
import grpc

import core_pb2 as core
import core_pb2_grpc as crpc

import dataflow_ext_pb2 as dfext
import dataflow_ext_pb2_grpc as dfrpc

from dsbox.server.controller.session_handler import Session

class DataflowExt(dfrpc.DataflowExtServicer):

    def DescribeDataflow(self, request, context):
        ok = True
        response = self._create_response("Dataflow description")

        session = Session.get(request.context.session_id)
        if session is None:
            response = self._create_response("Dataflow description", code="SESSION_UNKNOWN")
            ok = False

        pipeline = session.get_pipeline(request.pipeline_id)
        if pipeline is None:
            response = self._create_response("Dataflow description", code="INTERNAL")
            ok = False

        modules = []
        connections = []
        if ok:
            for i in range(0, len(pipeline.primitives)):
                primitive = pipeline.primitives[i]
                inputs = [dfext.DataflowDescription.Input(
                    name = "input_data",
                    type = "pandas.DataFrame"
                )]
                outputs = [dfext.DataflowDescription.Output(
                    name = "output_data",
                    type = "pandas.DataFrame"
                )]
                if primitive.task == "Modeling":
                    inputs.append(dfext.DataflowDescription.Input(
                        name = "input_labels",
                        type = "pandas.DataFrame"
                    ))

                modules.append(dfext.DataflowDescription.Module(
                    id = primitive.cls,
                    type = primitive.type,
                    label = primitive.name,
                    inputs = inputs,
                    outputs = outputs
                ))
                if i > 0:
                    prev_primitive = pipeline.primitives[i-1]
                    connections.append(dfext.DataflowDescription.Connection(
                        from_module_id = prev_primitive.cls,
                        from_output_name = "output_data",
                        to_module_id = primitive.cls,
                        to_input_name = "input_data"
                    ))
        return dfext.DataflowDescription(
            pipeline_id = request.pipeline_id,
            response_info = response,
            modules = modules,
            connections = connections
        )

    def GetDataflowResults(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def add_to_server(self, server):
        dfrpc.add_DataflowExtServicer_to_server(self, server)

    def _create_response(self, message, code="OK"):
        status = core.Status(code=core.StatusCode.Value(code), details=message)
        response = core.Response(status=status)
        return response
