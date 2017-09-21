import core_pb2 as core
import core_pb2_grpc as crpc

from dsbox.planner.event_handler import PlannerEventHandler

class GRPC_PlannerEventHandler(PlannerEventHandler):
    def __init__(self, session):
        self.session = session
        self.session_context = core.SessionContext(session_id = session.id)

    def StartedPlanning(self):
        pass

    def SubmittedPipeline(self, pipeline):
        response = self._create_response("Pipeline Submitted")
        progress = self._create_progress("SUBMITTED")
        self.session.add_pipeline(pipeline)
        result = core.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id
        )
        pipeline.planner_result = result;
        return result

    def RunningPipeline(self, pipeline):
        response = self._create_response("Pipeline Running")
        progress = self._create_progress("RUNNING")
        result = core.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id
        )
        pipeline.planner_result = result;
        return result

    def CompletedPipeline(self, pipeline, result):
        response = self._create_response("Pipeline Completed", "OK")
        progress = self._create_progress("COMPLETED")
        pipeline_info = None
        if result is None:
            response = self._create_response("Pipeline Failed", "UNKNOWN")
        else:
            pipeline_info = self._create_pipeline(self.session.controller.helper.metric.name, result)

        self.session.update_pipeline(pipeline)

        result = core.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id,
            pipeline_info = pipeline_info
        )
        pipeline.planner_result = result;
        return result;

    def EndedPlanning(self):
        pass

    def StartExecutingPipeline(self, pipeline):
        response = self._create_response("Pipeline Started Running", "OK")
        progress = self._create_progress("RUNNING")
        result = core.PipelineExecuteResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id
        )
        pipeline.test_result = result;
        return result

    def ExecutedPipeline(self, pipeline, result_uris):
        if len(result_uris) > 0:
            response = self._create_response("Pipeline Completed", "OK")
        else:
            response = self._create_response("Pipeline Failed to run", "INTERNAL")

        progress = self._create_progress("COMPLETED")
        result = core.PipelineExecuteResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id,
            result_uris = result_uris
        )
        pipeline.test_result = result;
        return result

    def _create_response(self, message, code="OK"):
        status = core.Status(code=core.StatusCode.Value('OK'), details=message)
        response = core.Response(status=status)

    def _create_progress(self, value):
        return core.Progress.Value(value)

    def _create_score(self, metric, value):
        return core.Score(metric = core.Metric.Value(metric), value=value)

    def _create_pipeline(self, metric, result):
        score = self._create_score(metric, result[1])
        # FIXME: Change output type to what is mentioned in request
        # FIXME: Set output result uris
        output = core.OutputType.Value("FILE")
        pipeline = core.Pipeline(
            output = output,
            predict_result_uris = [],
            scores = [score]
        )
        return pipeline
