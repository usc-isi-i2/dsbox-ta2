import os
import core_pb2 as core
import core_pb2_grpc as crpc

from dsbox.planner.event_handler import PlannerEventHandler

class GRPC_PlannerEventHandler(PlannerEventHandler):
    def __init__(self, session):
        self.session = session
        self.session_context = core.SessionContext(session_id = session.id)

    def StartedPlanning(self):
        pass

    def ProblemNotImplemented(self):
        response = self._create_response("Not Implemented", "UNIMPLEMENTED")
        result = core.PipelineCreateResult(
            response_info = response
        )
        return result

    def SubmittedPipeline(self, pipeline):
        response = self._create_response("Pipeline Submitted")
        progress = self._create_progress("SUBMITTED")
        self.session.add_pipeline(pipeline)
        result = core.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id
        )
        self.session.cache_planner_result(pipeline, result)
        return result

    def RunningPipeline(self, pipeline):
        response = self._create_response("Pipeline Running")
        progress = self._create_progress("RUNNING")
        self.session.update_pipeline(pipeline)
        result = core.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id
        )
        self.session.cache_planner_result(pipeline, result)
        return result

    def CompletedPipeline(self, pipeline, exec_pipeline):
        response = self._create_response("Pipeline Completed", "OK")
        progress = self._create_progress("COMPLETED")
        pipeline_info = None
        if exec_pipeline is None:
            response = self._create_response("Pipeline Failed", "INTERNAL")
        else:
            pipeline_info = self._create_pipeline_info(exec_pipeline)
            # Update session pipeline
            self.session.update_pipeline(exec_pipeline)


        result = core.PipelineCreateResult(
            response_info = response,
            progress_info = progress,
            pipeline_id = pipeline.id,
            pipeline_info = pipeline_info
        )
        self.session.cache_planner_result(pipeline, result)
        return result

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
        self.session.cache_test_result(pipeline, result)
        return result

    def ExecutedPipeline(self, pipeline):
        result_uris = []
        if pipeline.test_result is not None:
            resultfile = self.session.save_prediction_file(pipeline.test_result)
            result_uris.append(self._create_uri_from_path(resultfile))
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
        self.session.cache_test_result(pipeline, result)
        return result

    def _create_response(self, message, code="OK"):
        status = core.Status(code=core.StatusCode.Value(code), details=message)
        response = core.Response(status=status)
        return response

    def _create_progress(self, value):
        return core.Progress.Value(value)

    def _create_score(self, metric, value):
        return core.Score(metric = core.Metric.Value(metric), value=value)

    def _create_pipeline_info(self, pipeline):
        prediction_uris = []
        scores = []
        if pipeline.planner_result is not None:
            resultfile = self.session.save_prediction_file(pipeline.planner_result)
            prediction_uris.append(self._create_uri_from_path(resultfile))
            for metric in pipeline.planner_result.metric_values.keys():
                metric_value = pipeline.planner_result.metric_values[metric]
                scores.append(self._create_score(metric, metric_value))

        # FIXME: Change output type to what is mentioned in request
        output = core.OutputType.Value("FILE")
        pipeline = core.Pipeline(
            output = output,
            predict_result_uris = prediction_uris,
            scores = scores
        )
        return pipeline

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
