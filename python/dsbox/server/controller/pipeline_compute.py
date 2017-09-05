"""The Python implementation of the GRPC pipeline.PipelineComputeServicer server."""
import grpc

import pipeline_service_pb2 as ps
import pipeline_service_pb2_grpc as psrpc

from dsbox.server.controller.grpc_event_handler import GRPC_PlannerEventHandler
from dsbox.server.controller.session_handler import Session

from dsbox.planner.controller import Controller, Feature
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric

class PipelineCompute(psrpc.PipelineComputeServicer):

    def __init__(self, libdir):
        self.libdir = libdir

    def StartSession(self, request, context):
        """Session management
        """
        session = Session.new()
        session_context = ps.SessionContext(session_id = session.id)
        status = ps.Status(code=ps.StatusCode.Value('OK'), details="Session started")
        response = ps.Response(status=status)
        session_response = ps.SessionResponse(
            response_info=response,
            user_agent = request.user_agent,
            version = request.version,
            context = session_context)
        context.set_code(grpc.StatusCode.OK)
        return session_response

    def EndSession(self, request, context):
        Session.delete(request.session_id)
        status = ps.Status(code=ps.StatusCode.Value('OK'), details="Session ended")
        response = ps.Response(status=status)
        context.set_code(grpc.StatusCode.OK)
        return response

    def _create_path_from_uri(self, uri):
        import os
        try:
            from urllib import parse as urlparse
        except ImportError:
            import urlparse
        p = urlparse.urlparse(uri)
        return os.path.abspath(os.path.join(p.netloc, p.path))

    def CreatePipelines(self, request, context):
        session = Session.get(request.context.session_id)
        if session is None:
            yield None
            return

        # Get training and target features
        train_features = []
        target_features = []
        for tfeature in request.train_features:
            datadir = self._create_path_from_uri(tfeature.data_uri)
            featureid = tfeature.feature_id
            train_features.append(Feature(datadir, featureid))
        for tfeature in request.target_features:
            datadir = self._create_path_from_uri(tfeature.data_uri)
            featureid = tfeature.feature_id
            target_features.append(Feature(datadir, featureid))

        # Create the planning controller
        session.controller = Controller(train_features, target_features, self.libdir, session.outputdir)

        # Get Problem details
        session.controller.task_type = TaskType[ps.TaskType.Name(request.task)]
        if request.task_subtype is not None:
            session.controller.task_subtype = TaskSubType[ps.TaskSubtype.Name(request.task_subtype)]
        session.controller.output_type = ps.OutputType.Name(request.output)
        # FIXME: Need to use multiple metrics (not just one)
        session.controller.metric = Metric[ps.Metric.Name(request.metrics[0])]
        session.controller.metric_function = session.controller._get_metric_function(session.controller.metric)

        # Start planning
        session.controller.initialize_planners()
        for result in session.controller.start(GRPC_PlannerEventHandler(session)):
            yield result

    def ExecutePipeline(self, request, context):
        """Predict step - multiple results messages returned via GRPC streaming.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListPipelines(self, request, context):
        """Get pipelines already present in the session.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetCreatePipelineResults(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetExecutePipelineResults(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateProblemSchema(self, request, context):
        """Update problem schema
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def add_to_server(self, server):
        psrpc.add_PipelineComputeServicer_to_server(self, server)
