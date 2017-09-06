"""The Python implementation of the GRPC pipeline.PipelineComputeServicer server."""
import os
import os.path
import grpc
import uuid

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
        """Return filename from a file:// uri """
        # Python 2 and 3 compatible import of urlparse
        try:
            from urllib import parse as urlparse
        except ImportError:
            import urlparse
        p = urlparse.urlparse(uri)
        return os.path.abspath(os.path.join(p.netloc, p.path))

    def _create_uri_from_path(self, path):
        """Return file:// uri from a filename."""
        # Python 2 and 3 compatible import of urlparse
        try:
            from urllib import request as urlparse
        except ImportError:
            import urlparse
        path = os.path.abspath(path)
        if isinstance(path, str):
            path = path.encode('utf8')
        return 'file://' + urlparse.pathname2url(path)

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

        cutoff = request.max_pipelines
        # Create the planning controller
        session.controller = Controller(train_features, target_features, self.libdir, session.outputdir)
        session.controller.key = str(uuid.uuid1())

        # Get Problem details
        session.controller.task_type = TaskType[ps.TaskType.Name(request.task)]
        if request.task_subtype is not None:
            session.controller.task_subtype = TaskSubType[ps.TaskSubtype.Name(request.task_subtype)]
        session.controller.output_type = ps.OutputType.Name(request.output)

        # FIXME: Handle multiple metrics
        session.controller.metric = Metric[ ps.Metric.Name(request.metrics[0]) ]
        session.controller.metric_function = session.controller._get_metric_function(session.controller.metric)

        # Start planning
        session.controller.initialize_planners()
        for result in session.controller.train(GRPC_PlannerEventHandler(session), cutoff=cutoff):
            yield result

    def ExecutePipeline(self, request, context):
        """Predict step - multiple results messages returned via GRPC streaming.
        """
        session = Session.get(request.context.session_id)
        if session is None:
            yield None
            return

        pipeline = session.get_pipeline(request.pipeline_id)
        if pipeline is None:
            yield None
            return

        # Get test data directories
        handler = GRPC_PlannerEventHandler(session)
        handler.StartExecutingPipeline(pipeline)
        result_uris = []
        for data_uri in request.predict_dataset_uris:
            test_directory = self._create_path_from_uri(data_uri)
            resultfile = session.controller.test(pipeline, test_directory)
            result_uris.append(self._create_uri_from_path(resultfile))

        yield handler.ExecutedPipeline(pipeline, result_uris)


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
