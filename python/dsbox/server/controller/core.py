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
        if len(request.train_features) > 0:
            for tfeature in request.train_features:
                datadir = self._create_path_from_uri(tfeature.data_uri)
                featureid = tfeature.feature_id
                train_features.append(Feature(datadir, featureid))
        if len(request.target_features) > 0:
            for tfeature in request.target_features:
                datadir = self._create_path_from_uri(tfeature.data_uri)
                featureid = tfeature.feature_id
                target_features.append(Feature(datadir, featureid))

        # Create the planning controller if not already present
        c = session.controller
        if c is None:
            c = Controller(self.libdir)
            c.initialize_from_features(session.outputdir,
                train_features, target_features, view='TRAIN')

        # Set Problem schema
        if request.task > 0:
            c.problem.task_type = TaskType[core.TaskType.Name(request.task)]
        if request.task_subtype > 0:
            c.problem.task_subtype = TaskSubType[core.TaskSubtype.Name(request.task_subtype)]
        #if request.output > 0:
        #    c.problem.output_type = core.OutputType.Name(request.output)

        # Load metrics
        metrics = []
        if len(request.metrics) > 0:
            for rm in request.metrics:
                metrics.append( Metric[core.Metric.Name(rm)] )
            c.problem.set_metrics(metrics)

        # Set the max pipelines cutoff
        cutoff = None
        if request.max_pipelines is not None:
            cutoff = request.max_pipelines

        # Start planning / training
        session.controller = c
        if session.controller.l1_planner is None:
            session.controller.initialize_planners()

        for result in session.controller.train(GRPC_PlannerEventHandler(session), cutoff=cutoff):
            if result is not None:
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

        target_columns = session.controller.data_manager.target_columns
        target_data = session.controller.data_manager.target_data

        session.controller.initialize_from_features(session.outputdir, test_features, [], view='TEST')

        session.controller.data_manager.target_columns = target_columns
        session.controller.data_manager.target_data = target_data

        # Change this to be a yield too.. Save should happen within the handler
        for result in session.controller.test(pipeline, handler):
            if result is not None:
                yield result


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
                result = session.planner_results.get(pipeline_id, None)
                if result is not None:
                    yield result

    def GetExecutePipelineResults(self, request, context):
        session = Session.get(request.context.session_id)
        if session is not None:
            for pipeline_id in request.pipeline_ids:
                result = session.test_results.get(pipeline_id, None)
                if result is not None:
                    yield result

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
                if update.task_type:
                    # Get Problem details
                    c.problem.task_type = TaskType[core.TaskType.Name(update.task_type)]
                if update.task_subtype:
                    c.problem.task_subtype = TaskSubType[core.TaskSubtype.Name(update.task_subtype)]
                #if update.output_type:
                #    c.problem.output_type = core.OutputType.Name(update.output_type)
                if update.metric:
                    metric = Metric[core.Metric.Name(update.metric)]
                    c.problem.set_metrics([metric])
            session.controller = c
            return self._create_response("Updated Problem Schema")

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

    def _create_response(self, message, code="OK"):
        status = core.Status(code=core.StatusCode.Value(code), details=message)
        response = core.Response(status=status)
        return response
