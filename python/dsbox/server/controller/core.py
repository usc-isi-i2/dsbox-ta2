"""The Python implementation of the GRPC pipeline.CoreServicer server."""
import os
import os.path
import grpc
import uuid

import core_pb2 as core
import core_pb2_grpc as crpc

from dsbox.server.controller.grpc_event_handler import GRPC_PlannerEventHandler
from dsbox.server.controller.session_handler import Session

from dsbox.planner.controller import Controller, Feature
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric

class Core(crpc.CoreServicer):

    def __init__(self, libdir):
        self.libdir = libdir

    def StartSession(self, request, context):
        """Session management
        """
        session = Session.new()
        session_context = core.SessionContext(session_id = session.id)
        response = self._create_response("Session started")
        session_response = core.SessionResponse(
            response_info=response,
            user_agent = request.user_agent,
            version = request.version,
            context = session_context)
        return session_response

    def EndSession(self, request, context):
        Session.delete(request.session_id)
        return self._create_response("Session ended")

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
        c = Controller(self.libdir)
        c.set_config_simple('', session.outputdir)
        c.initialize_data_from_features(train_features, target_features)

        # Get Problem details
        c.helper.task_type = TaskType[core.TaskType.Name(request.task)]
        if request.task_subtype is not None:
            c.task_subtype = TaskSubType[core.TaskSubtype.Name(request.task_subtype)]
        c.output_type = core.OutputType.Name(request.output)

        # Load metrics
        metrics = []
        for rm in request.metrics:
            metrics.append( Metric[core.Metric.Name(rm)] )
        c.helper.set_metrics(metrics)

        # Start planning
        session.controller = c
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

        # Get test features
        test_features = []
        for tfeature in request.predict_features:
            datadir = self._create_path_from_uri(tfeature.data_uri)
            featureid = tfeature.feature_id
            test_features.append(Feature(datadir, featureid))

        # Get test data directories
        handler = GRPC_PlannerEventHandler(session)
        handler.StartExecutingPipeline(pipeline)
        result_uris = []

        session.controller.initialize_test_data_from_features(test_features)

        resultfile = session.controller.test(pipeline)
        if resultfile is not None:
            result_uris.append(self._create_uri_from_path(resultfile))

        yield handler.ExecutedPipeline(pipeline, result_uris)

    def ListPipelines(self, request, context):
        pipeline_ids = []
        session = Session.get(request.context.session_id)
        if session is not None:
            for pipeline_id in session.pipelines.keys():
                pipeline_ids.append(pipeline_id)
        response = self._create_response("Listing pipelines")
        return core.PipelineListResult(
            response_info = response,
            pipeline_ids = pipeline_ids
        )


    def DeletePipelines(self, request, context):
        pipeline_ids = []
        session = Session.get(request.context.session_id)
        if session is not None:
            for pipeline_id in request.delete_pipeline_ids:
                session.delete_pipeline(pipeline_id)
                pipeline_ids.append(pipeline_id)
        response = self._create_response("Deleted pipelines")
        return core.PipelineListResult(
            response_info = response,
            pipeline_ids = pipeline_ids
        )

    def GetCreatePipelineResults(self, request, context):
        session = Session.get(request.context.session_id)
        if session is not None:
            for pipeline_id in request.pipeline_ids:
                pipeline = session.get_pipeline(pipeline_id)
                if pipeline is not None and pipeline.planner_result is not None:
                    yield pipeline.planner_result

    def GetExecutePipelineResults(self, request, context):
        session = Session.get(request.context.session_id)
        if session is not None:
            for pipeline_id in request.pipeline_ids:
                pipeline = session.get_pipeline(pipeline_id)
                if pipeline is not None and pipeline.test_result is not None:
                    yield pipeline.test_result

    def ExportPipeline(self, request, context):
        """Export executable of a pipeline, including any optional preprocessing used in session
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateProblemSchema(self, request, context):
        session = Session.get(request.context.session_id)
        if session is not None:
            c = session.controller
            for update in request.updates:
                if update.task_type is not None:
                    # Get Problem details
                    c.helper.task_type = TaskType[core.TaskType.Name(update.task_type)]
                if update.task_subtype is not None:
                    c.task_subtype = TaskSubType[core.TaskSubtype.Name(update.task_subtype)]
                if update.output_type is not None:
                    c.output_type = core.OutputType.Name(update.output_type)
                if update.metric is not None:
                    # FIXME: Handle multiple metrics
                    c.helper.metric = Metric[ core.Metric.Name(update.metric) ]
                    c.helper.metric_function = c.helper._get_metric_function(c.helper.metric)
                return core.PipelineListResult(
                    response_info = self._create_response("Updated Problem Schema")
                )

    def add_to_server(self, server):
        crpc.add_CoreServicer_to_server(self, server)


    '''
    Helper functions
    '''
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
        # Python 2 and 3 compatible import of urllib
        try:
            from urllib import request as urllib
        except ImportError:
            import urllib
        path = os.path.abspath(path)
        if isinstance(path, str):
            path = path.encode('utf8')
        return 'file://' + urllib.pathname2url(path)

    def _create_response(self, message, code="OK"):
        status = core.Status(code=core.StatusCode.Value(code), details=message)
        response = core.Response(status=status)
        return response
