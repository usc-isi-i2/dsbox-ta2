import sys
sys.path.append('/Users/minazuki/Desktop/studies/master/2018Summer/DSBOX_new/dsbox-ta2/python')

from dsbox.pipeline.fitted_pipeline import FittedPipeline
# from dsbox.datapostprocessing.vertical_concat import VerticalConcat, EnsembleVoting
from d3m import runtime as runtime_module, container
from d3m.metadata import pipeline as pipeline_module
import d3m.primitives
import d3m.exceptions as exceptions
import copy
from d3m.metadata.base import ALL_ELEMENTS

def set_target_column(dataset):
    for index in range(
            dataset.metadata.query(('0', ALL_ELEMENTS))['dimension']['length'] - 1,
            -1, -1):
        column_semantic_types = dataset.metadata.query(
            ('0', ALL_ELEMENTS, index))['semantic_types']
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in \
                column_semantic_types:
            column_semantic_types = list(column_semantic_types) + [
                'https://metadata.datadrivendiscovery.org/types/Target',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget']
            dataset.metadata = dataset.metadata.update(
                ('0', ALL_ELEMENTS, index), {'semantic_types': column_semantic_types})
            return

    raise exceptions.InvalidArgumentValueError(
        'At least one column should have semantic type SuggestedTarget')


pipelines_dir = '/Users/minazuki/Desktop/studies/master/2018Summer/data'
log_dir = '/Users/minazuki/Desktop/studies/master/2018Summer/data/log'
step_outputs = []
voting_pipeline = pipeline_module.Pipeline('voting', context=pipeline_module.PipelineContext.TESTING)
pipeline_input = voting_pipeline.add_input(name='inputs')

pids = ['3c5f6bfa-4d3b-43b4-a371-af7be9e2a938','bcdab3e5-eb82-438c-83cb-d7f851754536',
        'c0b4d68e-16ca-4042-9d51-8df706a7ecf1','7370ab30-c7cf-461e-a94b-f7a50ebbaaf0']

for each_pid in pids:
    each_dsbox_fitted, each_runtime = FittedPipeline.load(pipelines_dir, each_pid, log_dir)
    each_fitted = runtime_module.FittedPipeline(each_pid, each_runtime, context=pipeline_module.PipelineContext.TESTING)
    each_step = pipeline_module.FittedPipelineStep(each_fitted.id, each_fitted)
    each_step.add_input(pipeline_input)
    voting_pipeline.add_step(each_step)
    step_outputs.append(each_step.add_output('output'))

concat_step = pipeline_module.PrimitiveStep({
    "python_path": "d3m.primitives.dsbox.VerticalConcat",
    "id": "dsbox-vertical-concat",
    "version": "1.3.0",
    "name": "DSBox vertically concat"})

for i in range(len(pids) - 1):
    each_concact_step = copy.deepcopy(concat_step)
    if i == 0:
        each_concact_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_outputs[i])
    else:
        each_concact_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
    each_concact_step.add_argument(name='inputs2', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_outputs[i+1])

    voting_pipeline.add_step(each_concact_step)
    # update concat_step_output
    concat_step_output = each_concact_step.add_output('produce')

vote_step = pipeline_module.PrimitiveStep({
    "python_path": "d3m.primitives.dsbox.EnsembleVoting",
    "id": "dsbox-ensemble-voting",
    "version": "1.3.0",
    "name": "DSBox ensemble voting"})

vote_step.add_argument(name='inputs', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
voting_pipeline.add_step(vote_step)
voting_output = vote_step.add_output('produce')

voting_pipeline.add_output(name='Metafeatures', data_reference=voting_output)

tpp = FittedPipeline(pipeline = voting_pipeline, dataset_id = '38_sick', log_dir = log_dir, metric_descriptions = "no")
# rr = runtime_module.Runtime(voting_pipeline)

dataset = container.Dataset.load('file:///Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/38_sick_dataset/datasetDoc.json')
set_target_column(dataset)

tpp.runtime.set_not_use_cache()
tpp.fit(inputs = [dataset])
# import pdb
# pdb.set_trace()
tpp.produce(inputs = [dataset])
