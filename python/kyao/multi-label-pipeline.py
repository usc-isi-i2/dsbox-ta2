import typing

from d3m import index
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata import base as metadata_base
from d3m.container.dataset import get_dataset

# from d3m.runtime import Runtime
from dsbox.template.runtime import Runtime
from dsbox.controller.config import DsboxConfig

'''
export D3MRUN=ta2
export D3MINPUTDIR=/home/ktyao/data/datasets-v32/seed_datasets_current/LL1_multilearn_emotions
export D3MOUTPUTDIR=/home/ktyao/dsbox/output/seed/LL1_multilearn_emotions
export D3MLOCALDIR=/home/ktyao/dsbox/output/seed/LL1_multilearn_emotions/tmp
export D3MSTATICDIR=/home/ktyao/dsbox/static
export D3MPROBLEMPATH=/home/ktyao/data/datasets-v32/seed_datasets_current/LL1_multilearn_emotions/TRAIN/problem_TRAIN/problemDoc.json
export D3MCPU=2
export D3MRAM=50
export D3MTIMEOUT=600
export DSBOXTESTDATASETID=LL1_multilearn_emotions_dataset_TEST
'''
config = DsboxConfig()
config.load()

primitive_list = index.search()


def add_step(pipeline: Pipeline, primitive_name: str, inputs: typing.List[int]):
    primitive_description = index.get_primitive(primitive_name).metadata.query()
    step = PrimitiveStep(primitive_description=primitive_description)
    for name, input_step in zip(['inputs', 'outputs'], inputs):
        if input_step < 0:
            ref = 'inputs.0'
        else:
            ref = f'steps.{input_step}.produce'
        step.add_argument(name=name, argument_type=metadata_base.ArgumentType.CONTAINER, data_reference=ref)
    step.add_output('produce')
    pipeline.add_step(step)
    return len(pipeline.steps) - 1


# Creating Pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

denormalize = add_step(pipeline_description, 'd3m.primitives.data_transformation.denormalize.Common', [-1])
to_dataframe = add_step(pipeline_description, 'd3m.primitives.data_transformation.dataset_to_dataframe.Common', [denormalize])

get_attributes = add_step(pipeline_description, 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon', [to_dataframe])
pipeline_description.steps[get_attributes].add_hyperparameter(name='semantic_types',
                                                              argument_type=metadata_base.ArgumentType.VALUE,
                                                              data=[
                                                                  "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
                                                                  "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
                                                                  "https://metadata.datadrivendiscovery.org/types/Attribute"
                                                              ])

get_target = add_step(pipeline_description, 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon', [to_dataframe])
pipeline_description.steps[get_target].add_hyperparameter(name='semantic_types',
                                                          argument_type=metadata_base.ArgumentType.VALUE,
                                                          data=[
                                                              "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
                                                              "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
                                                              "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"
                                                          ])

parsed = add_step(pipeline_description, 'd3m.primitives.data_transformation.column_parser.DataFrameCommon', [get_attributes])

multilable_classifier = add_step(pipeline_description, 'd3m.primitives.classification.multilabel_classifier.DSBOX',
                                 [parsed, get_target])

pipeline_description.add_output(f'steps.{multilable_classifier}.produce', name='outputs.0')

bibtex = False
if bibtex:
    dataset = get_dataset('/c/Users/ktyao/data/datasets-v32/seed_datasets_current/uu11_bibtex/uu11_bibtex_dataset/datasetDoc.json')
    target = {
        'resource_id': 'learningData',
        'column_index': 1837
    }
else:
    # dataset = get_dataset('/c/Users/ktyao/data/datasets-v32/seed_datasets_current/LL1_multilearn_emotions/TRAIN/dataset_TRAIN/datasetDoc.json')
    dataset = get_dataset('/lfs1/ktyao/DSBox/data/datasets/seed_datasets_current/LL1_multilearn_emotions/TRAIN/dataset_TRAIN/datasetDoc.json')
    target = {
        'resource_id': 'learningData',
        'column_index': 73
    }


inputs = [dataset]
dataset.metadata = dataset.metadata.add_semantic_type(
    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
)
dataset.metadata = dataset.metadata.remove_semantic_type(
    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
    'https://metadata.datadrivendiscovery.org/types/Attribute',
)


runtime = Runtime(pipeline_description, context=metadata_base.Context.TESTING, log_dir=config.log_dir)
fit_result = runtime.fit(inputs, return_values=['outputs.0'])
produce_result = runtime.produce(inputs, return_values=['outputs.0'])
