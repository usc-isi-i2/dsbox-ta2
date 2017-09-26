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
        session = Session.get(request.context.session_id)
        ok = True
        if session is None:
            response = self._create_response("Dataflow results", code="SESSION_UNKNOWN")
            ok = False

        pipeline = session.get_pipeline(request.pipeline_id)
        if pipeline is None:
            response = self._create_response("Dataflow results", code="INTERNAL")
            ok = False
        if ok:
            if pipeline.finished:
                for result in self._get_pipeline_results(pipeline):
                    yield result
            else:
                while not pipeline.finished:
                    pipeline.waitForChanges()
                    for result in self._get_pipeline_results(pipeline):
                        yield result
        else:
            yield dfext.ModuleResult(response_info = response)

    def _get_pipeline_results(self, pipeline):
        response = self._create_response("Dataflow results")
        for primitive in pipeline.primitives:
            status = dfext.ModuleResult.PENDING
            if primitive.start_time is not None:
                status = dfext.ModuleResult.Status.Value('RUNNING')
            if primitive.finished:
                if primitive.progress == 1.0:
                    status = dfext.ModuleResult.Status.Value('DONE')
                else:
                    status = dfext.ModuleResult.Status.Value('ERROR')

            execution_time = None
            if primitive.end_time is not None:
                excution_time = primitive.end_time - primitive.start_time
            result = dfext.ModuleResult(
                response_info = response,
                module_id = primitive.cls,
                progress = primitive.progress,
                execution_time = execution_time,
                status = status
            )
            yield result

    def add_to_server(self, server):
        dfrpc.add_DataflowExtServicer_to_server(self, server)

    def _create_response(self, message, code="OK"):
        status = core.Status(code=core.StatusCode.Value(code), details=message)
        response = core.Response(status=status)
        return response
