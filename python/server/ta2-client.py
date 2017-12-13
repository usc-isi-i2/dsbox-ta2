#!/usr/bin/env python

import sys
import os.path

# Setup Paths
PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENTDIR)

from dsbox_dev_setup import path_setup
path_setup()

import grpc
import urllib
import pandas

import core_pb2 as core
import core_pb2_grpc as crpc
import data_ext_pb2 as dext
import data_ext_pb2_grpc as drpc
import dataflow_ext_pb2 as dfext
import dataflow_ext_pb2_grpc as dfrpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = crpc.CoreStub(channel)
    dstub = drpc.DataExtStub(channel)
    dfstub = dfrpc.DataflowExtStub(channel)

    # Start Session
    session_response = stub.StartSession(
        core.SessionRequest(user_agent="xxx", version="1.0"))
    session_context = session_response.context
    print("Session started (%s)" % str(session_context.session_id))

    # Send pipeline creation request
    all_features = [
        core.Feature(
            data_uri="file:///tmp/data/185_baseball/185_baseball_dataset", feature_id="*")
    ]
    some_features = [
        core.Feature(data_uri="file:///tmp/data/185_baseball/185_baseball_dataset", feature_id="d3mIndex"),
        core.Feature(data_uri="file:///tmp/data/185_baseball/185_baseball_dataset", feature_id="Games_played"),
        core.Feature(data_uri="file:///tmp/data/185_baseball/185_baseball_dataset", feature_id="Runs"),
        core.Feature(data_uri="file:///tmp/data/185_baseball/185_baseball_dataset", feature_id="Hits"),
        core.Feature(data_uri="file:///tmp/data/185_baseball/185_baseball_dataset", feature_id="Home_runs")
    ]
    target_features = [
        core.Feature(data_uri="file:///tmp/data/185_baseball/185_baseball_dataset", feature_id="Hall_of_Fame")
    ]
    task = core.TaskType.Value('CLASSIFICATION')
    task_subtype = core.TaskSubtype.Value('MULTICLASS')
    task_description = "Classify Hall of Fame"
    output = core.OutputType.Value('FILE')
    metrics = [core.Metric.Value('F1_MICRO'), core.Metric.Value('F1_MACRO')]
    max_pipelines = 10

    pipeline_ids = []

    print("Training with some features")
    pc_request = core.PipelineCreateRequest(
        context=session_context,
        train_features=some_features,
        task=task,
        task_subtype=task_subtype,
        task_description=task_description,
        output=output,
        metrics=metrics,
        target_features=target_features,
        max_pipelines=max_pipelines
    )

    '''
    # Iterate over results
    for pcr in stub.CreatePipelines(pc_request):
        print(str(pcr))
        if len(pcr.pipeline_info.scores) > 0:
            pipeline_ids.append(pcr.pipeline_id)

    print("Training with some features")
    pc_request = core.PipelineCreateRequest(
        context = session_context,
        train_features = some_features,
        task = task,
        task_subtype = task_subtype,
        task_description = task_description,
        output = output,
        metrics = metrics,
        target_features = target_features,
        max_pipelines = max_pipelines
    )
    '''

    # Iterate over results
    for pcr in stub.CreatePipelines(pc_request):
        print(str(pcr))
        '''
        for gdr in dfstub.GetDataflowResults(dfext.PipelineReference(context = session_context,
                pipeline_id = pcr.pipeline_id)):
            print(gdr)
        '''
        if len(pcr.pipeline_info.scores) > 0:
            pipeline_ids.append(pcr.pipeline_id)
            dflow = dfstub.DescribeDataflow(dfext.PipelineReference(
                context = session_context,
                pipeline_id = pcr.pipeline_id
            ))
            print(dflow)
            '''
            if len(pcr.pipeline_info.predict_result_uris) > 0:
                df = pandas.read_csv(pcr.pipeline_info.predict_result_uris[0], index_col="d3mIndex")
                print(df)
            '''

    print ("************** Executing/Testing Pipelines")

    # Execute pipelines
    for pipeline_id in pipeline_ids:
        print("Executing Pipeline %s" % pipeline_id)
        ep_request = core.PipelineExecuteRequest(
            context=session_context,
            pipeline_id=pipeline_id,
            predict_features=some_features
        )
        for ecr in stub.ExecutePipeline(ep_request):
            print(str(ecr))
            if len(ecr.result_uris) > 0:
                df = pandas.read_csv(ecr.result_uris[0], index_col="d3mIndex")
                print(df)

    list_request = core.PipelineListRequest(context=session_context)
    lrr = stub.ListPipelines(list_request)
    print (lrr.pipeline_ids)

    print ("************** Cached pipeline create results")
    pcrr = core.PipelineCreateResultsRequest(
        context = session_context,
        pipeline_ids = lrr.pipeline_ids
    )
    for gcpr in stub.GetCreatePipelineResults(pcrr):
        print(str(gcpr))

    print ("************** Cached pipeline execute results")
    perr = core.PipelineExecuteResultsRequest(
        context = session_context,
        pipeline_ids = lrr.pipeline_ids
    )
    for gepr in stub.GetExecutePipelineResults(perr):
        print(str(gepr))

    print ("*********** Updating Metric to Accuracy.. Create pipelines again")
    metric = core.Metric.Value('ACCURACY')
    ups_request = core.UpdateProblemSchemaRequest(
        context=session_context,
        updates=[
            core.UpdateProblemSchemaRequest.ReplaceProblemSchemaField(metric=metric)
        ]
    )
    stub.UpdateProblemSchema(ups_request)
    for pcr in stub.CreatePipelines(core.PipelineCreateRequest(context=session_context)):
        print(str(pcr))

    stub.EndSession(session_context)


if __name__ == '__main__':
    run()
