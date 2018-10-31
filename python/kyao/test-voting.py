import sys
sys.path.append('/nfs1/dsbox-repo/kyao/dsbox-devel/dsbox-ta2/python')

from dsbox.pipeline.fitted_pipeline import FittedPipeline
# from dsbox.datapostprocessing.vertical_concat import VerticalConcat, EnsembleVoting
from d3m import runtime as runtime_module, container
from d3m.metadata import pipeline as pipeline_module
import d3m.primitives
import d3m.exceptions as exceptions
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


pipelines_dir = '/nas/home/kyao/dsbox/run-july/seed/38_sick'
log_dir = '/nas/home/kyao/dsbox/run-july/seed/38_sick/log'
pid0 = '0ecd7d2e-4620-430a-881e-b93ff0cfb692'
pid1 = '468d36d6-29d3-4cbc-8b0a-48bbebe50f68'
pid2 = '677ed24c-0e09-476f-a299-b71a7a5afdfc'

dsbox_fitted0, runtime0 = FittedPipeline.load(pipelines_dir, pid0, log_dir)
dsbox_fitted1, runtime1 = FittedPipeline.load(pipelines_dir, pid1, log_dir)
dsbox_fitted2, runtime2 = FittedPipeline.load(pipelines_dir, pid2, log_dir)

fitted0 = runtime_module.FittedPipeline('fp0', runtime0, context=pipeline_module.PipelineContext.TESTING)
fitted1 = runtime_module.FittedPipeline('fp1', runtime1, context=pipeline_module.PipelineContext.TESTING)
fitted2 = runtime_module.FittedPipeline('fp2', runtime2, context=pipeline_module.PipelineContext.TESTING)


voting_pipeline = pipeline_module.Pipeline('voting', context=pipeline_module.PipelineContext.TESTING)
pipeline_input = voting_pipeline.add_input(name='inputs')

step_0 = pipeline_module.FittedPipelineStep(fitted0.id, fitted0)
step_0.add_input(pipeline_input)
voting_pipeline.add_step(step_0)
step_0_output = step_0.add_output('output')

step_1 = pipeline_module.FittedPipelineStep(fitted1.id, fitted1)
step_1.add_input(pipeline_input)
voting_pipeline.add_step(step_1)
step_1_output = step_1.add_output('output')

step_2 = pipeline_module.FittedPipelineStep(fitted2.id, fitted2)
step_2.add_input(pipeline_input)
voting_pipeline.add_step(step_2)
step_2_output = step_2.add_output('output')

concat_step = pipeline_module.PrimitiveStep({
    "python_path": "d3m.primitives.dsbox.VerticalConcat",
    "id": "dsbox-vertical-concat",
    "version": "1.3.0",
    "name": "DSBox vertically concat"})
concat_step.add_argument(name='inputs', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_0_output)
concat_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_1_output)
concat_step.add_argument(name='inputs2', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_2_output)
voting_pipeline.add_step(concat_step)
concat_step_output = concat_step.add_output('produce')



vote_step = pipeline_module.PrimitiveStep({
    "python_path": "d3m.primitives.dsbox.EnsembleVoting",
    "id": "dsbox-ensemble-voting",
    "version": "1.3.0",
    "name": "DSBox ensemble voting"})
vote_step.add_argument(name='inputs', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
voting_pipeline.add_step(vote_step)
voting_output = vote_step.add_output('produce')

voting_pipeline.add_output(name='Metafeatures', data_reference=voting_output)


runtime = runtime_module.Runtime(voting_pipeline)

dataset = container.Dataset.load('file:///nfs1/dsbox-repo/data/datasets/seed_datasets_current/38_sick/38_sick_dataset/datasetDoc.json')
set_target_column(dataset)

runtime.fit([dataset])
