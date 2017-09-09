import pipeline_service_pb2 as ps
import pipeline_service_pb2_grpc as psrpc

from dsbox.planner.event_handler import PlannerEventHandler

class GRPC_PlannerEventHandler(PlannerEventHandler):
    def __init__(self, session):
        self.session = session
        self.session_context = ps.SessionContext(session_id = session.id)

    def StartedPlanning(self):
        pass

    def SubmittedPipeline(self, pipeline):
        response = self._create_response("Pipeline Submitted")
        progress = self._create_progress("SUBMITTED")
        self.session.add_pipeline(pipeline)
        result = ps.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id
        )
        return result

    def RunningPipeline(self, pipeline):
        response = self._create_response("Pipeline Running")
        progress = self._create_progress("RUNNING")
        result = ps.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id
        )
        return result

    def CompletedPipeline(self, pipeline, result):
        response = self._create_response("Pipeline Completed", "OK")
        progress = self._create_progress("COMPLETED")
        pipeline_info = None
        if result is None:
            response = self._create_response("Pipeline Failed", "UNKNOWN")
        else:
            pipeline_info = self._create_pipeline(self.session.controller.metric.name, result)
        return ps.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id,
            pipeline_info = pipeline_info
        )

    def EndedPlanning(self):
        pass

    def StartExecutingPipeline(self, pipeline):
        response = self._create_response("Pipeline Started Running", "OK")
        progress = self._create_progress("RUNNING")
        return ps.PipelineExecuteResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id
        )

    def ExecutedPipeline(self, pipeline, result_uris):
        response = self._create_response("Pipeline Completed", "OK")
        progress = self._create_progress("COMPLETED")
        return ps.PipelineExecuteResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id,
            result_uris = result_uris
        )

    def _create_response(self, message, code="OK"):
        status = ps.Status(code=ps.StatusCode.Value('OK'), details=message)
        response = ps.Response(status=status)

    def _create_progress(self, value):
        return ps.Progress.Value(value)

    def _create_score(self, metric, value):
        return ps.Score(metric = ps.Metric.Value(metric), value=value)

    def _create_pipeline(self, metric, result):
        score = self._create_score(metric, result[1])
        # FIXME: Change output type to what is mentioned in request
        # FIXME: Set output result uris
        output = ps.OutputType.Value("FILE")
        pipeline = ps.Pipeline(
            output = output,
            predict_result_uris = [],
            scores = [score]
        )
        return pipeline
