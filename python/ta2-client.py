"""The Python implementation of the GRPC pipeline.PipelineCompute client."""

import grpc

import pipeline_service_pb2 as ps
import pipeline_service_pb2_grpc as psrpc


def run():
  channel = grpc.insecure_channel('localhost:50051')
  stub = psrpc.PipelineComputeStub(channel)

  # Start Session
  session_response = stub.StartSession(ps.SessionRequest(user_agent="xxx", version="1.0"))
  session_context = session_response.context
  print("Session started (%s)" % str(session_context.session_id))

  # Send pipeline creation request
  # FIXME: Only using the data_uri in the first train_features for now
  # - Everything else is gotten from the problemSchema
  # - Later we shall replace the problem schema with items from the request
  # - Then the data uri will point to the problem directory, problem's data
  #     directory , or the data itself ?
  # - If it is to the problem directory (i.e. with the problem schema, then why
  #     do we have these other attributes in the request)
  # - If to the data itself, then what about the data schema ?
  # - Currently there is a separate file for trainTargets,
  #   - Will the target_features change it ?

  train_features = [
    ps.Feature(data_uri="file:///Users/Varun/git/dsbox/data/o_185/data", feature_id="*")
    ]
  train_features_some = [
    ps.Feature(data_uri="file:///Users/Varun/git/dsbox/data/o_185/data", feature_id="Games_played"),
    ps.Feature(data_uri="file:///Users/Varun/git/dsbox/data/o_185/data", feature_id="Runs"),
    ps.Feature(data_uri="file:///Users/Varun/git/dsbox/data/o_185/data", feature_id="Hits"),
    ps.Feature(data_uri="file:///Users/Varun/git/dsbox/data/o_185/data", feature_id="Home_runs")
    ]
  task = ps.TaskType.Value('CLASSIFICATION')
  task_subtype = ps.TaskSubtype.Value('MULTICLASS')
  task_description = "Classify Hall of Fame"
  output = ps.OutputType.Value('FILE')
  metrics = [ps.Metric.Value('F1_MICRO')]
  target_features = [
    ps.Feature(data_uri="file:///Users/Varun/git/dsbox/data/o_185/data", feature_id="*")
    ]
  max_pipelines = 20

  print("Training with all features")
  pc_request = ps.PipelineCreateRequest(
    context = session_context,
    train_features = train_features,
    task = task,
    task_subtype = task_subtype,
    task_description = task_description,
    output = output,
    metrics = metrics,
    target_features = target_features,
    max_pipelines = max_pipelines
  )

  # Iterate over results
  for pcr in stub.CreatePipelines(pc_request):
      print(str(pcr))

  print("Training with some features")
  pc_request = ps.PipelineCreateRequest(
    context = session_context,
    train_features = train_features_some,
    task = task,
    task_subtype = task_subtype,
    task_description = task_description,
    output = output,
    metrics = metrics,
    target_features = target_features,
    max_pipelines = max_pipelines
  )

  # Iterate over results
  for pcr in stub.CreatePipelines(pc_request):
      print(str(pcr))

  stub.EndSession(session_context)

if __name__ == '__main__':
  run()
