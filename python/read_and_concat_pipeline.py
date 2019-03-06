import sys
sys.path.append('/Users/muxin/Desktop/ISI/dsbox_env/dsbox-ta2/python')

from dsbox.pipeline.fitted_pipeline import FittedPipeline
# from dsbox.datapostprocessing.vertical_concat import VerticalConcat, EnsembleVoting
from d3m import runtime as runtime_module, container
from d3m.metadata import pipeline as pipeline_module
import d3m.primitives
import d3m.exceptions as exceptions
from d3m.metadata.base import ALL_ELEMENTS
from d3m import index as d3m_index

def preprocessing_pipeline():
    preprocessing_pipeline = pipeline_module.Pipeline('big', context=pipeline_module.PipelineContext.TESTING)
    initial_input = preprocessing_pipeline.add_input(name="inputs")
    denormalize_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Denormalize").metadata.query()))
    denormalize_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=initial_input)
    preprocessing_pipeline.add_step(denormalize_step)
    denormalize_step_output = denormalize_step.add_output('produce')
    to_dataframe_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.datasets.DatasetToDataFrame").metadata.query()))
    to_dataframe_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=denormalize_step_output)
    preprocessing_pipeline.add_step(to_dataframe_step)
    to_dataframe_step_output = to_dataframe_step.add_output("produce")
    extract_attribute_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.ExtractColumnsBySemanticTypes").metadata.query()))
    extract_attribute_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=to_dataframe_step_output)
    preprocessing_pipeline.add_step(extract_attribute_step)
    extract_attribute_step_output = extract_attribute_step.add_output("produce")
    extract_attribute_step.add_hyperparameter(name='semantic_types',argument_type=pipeline_module.ArgumentType.VALUE, data=(
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                                    ))
    profiler_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Profiler").metadata.query()))
    profiler_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=extract_attribute_step_output)
    preprocessing_pipeline.add_step(profiler_step)
    profiler_step_output = profiler_step.add_output("produce")
    clean_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.CleaningFeaturizer").metadata.query()))
    clean_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=profiler_step_output)
    preprocessing_pipeline.add_step(clean_step)
    clean_step_output = clean_step.add_output("produce")
    corex_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.CorexText").metadata.query()))
    corex_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=clean_step_output)
    preprocessing_pipeline.add_step(corex_step)
    corex_step_output = corex_step.add_output("produce")
    encoder_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Encoder").metadata.query()))
    encoder_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=corex_step_output)
    preprocessing_pipeline.add_step(encoder_step)
    encoder_step_output = encoder_step.add_output("produce")
    impute_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.MeanImputation").metadata.query()))
    impute_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=encoder_step_output)
    preprocessing_pipeline.add_step(impute_step)
    impute_step_output = impute_step.add_output("produce")
    scalar_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.IQRScaler").metadata.query()))
    scalar_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=impute_step_output)
    preprocessing_pipeline.add_step(scalar_step)
    scalar_step_output = scalar_step.add_output("produce")
    extract_target_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.ExtractColumnsBySemanticTypes").metadata.query()))
    extract_target_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=to_dataframe_step_output)
    preprocessing_pipeline.add_step(extract_target_step)
    extract_target_step_output = extract_target_step.add_output("produce")
    extract_target_step.add_hyperparameter(name='semantic_types',argument_type=pipeline_module.ArgumentType.VALUE, data=(
                                              'https://metadata.datadrivendiscovery.org/types/Target',
                                              'https://metadata.datadrivendiscovery.org/types/TrueTarget'
                                              ))
    # preprocessing_pipeline.add_output(name="produce", data_reference=scalar_step_output)
    return preprocessing_pipeline, scalar_step_output, initial_input, extract_target_step_output





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


pipelines_dir = '/Users/muxin/Desktop/ISI/dsbox-env/output/seed/38_sick/'
log_dir = '/Users/muxin/Desktop/studies/master/2018Summer/data/log'
pid0 = '32b24d72-44c6-4956-bc21-835cb42f0f2e'
pid1 = 'a8f4001a-64f4-4ff1-a89d-3548f4dfeb88'
pid2 = '5e1d9723-ec02-46d2-abdf-46389fba8e52'

dsbox_fitted0 = FittedPipeline.load(folder_loc=pipelines_dir, fitted_pipeline_id=pid0, log_dir=log_dir)
runtime0 = dsbox_fitted0.runtime
dsbox_fitted1 = FittedPipeline.load(folder_loc=pipelines_dir, fitted_pipeline_id=pid1, log_dir=log_dir)
runtime1 = dsbox_fitted1.runtime
dsbox_fitted2 = FittedPipeline.load(folder_loc=pipelines_dir, fitted_pipeline_id=pid2, log_dir=log_dir)
runtime2 = dsbox_fitted2.runtime

fitted0 = runtime_module.FittedPipeline(pid0, runtime0, context=pipeline_module.PipelineContext.TESTING)
fitted1 = runtime_module.FittedPipeline(pid1, runtime1, context=pipeline_module.PipelineContext.TESTING)
fitted2 = runtime_module.FittedPipeline(pid2, runtime2, context=pipeline_module.PipelineContext.TESTING)


# big_pipeline = pipeline_module.Pipeline('voting', context=pipeline_module.PipelineContext.TESTING)
# pipeline_input = big_pipeline.add_input(name='inputs')
big_pipeline, pipeline_output, pipeline_input, target = preprocessing_pipeline()



step_0 = pipeline_module.FittedPipelineStep(fitted0.id, fitted0)
step_0.add_input(pipeline_input)
big_pipeline.add_step(step_0)
step_0_output = step_0.add_output('output')




step_1 = pipeline_module.FittedPipelineStep(fitted1.id, fitted1)
step_1.add_input(pipeline_input)
big_pipeline.add_step(step_1)
step_1_output = step_1.add_output('output')

step_2 = pipeline_module.FittedPipelineStep(fitted2.id, fitted2)
step_2.add_input(pipeline_input)
big_pipeline.add_step(step_2)
step_2_output = step_2.add_output('output')

concat_step = pipeline_module.PrimitiveStep({
    "python_path": "d3m.primitives.dsbox.HorizontalConcat",
    "id": "dsbox-horizontal-concat",
    "version": "1.3.0",
    "name": "DSBox horizontal concat"
    })
concat_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_0_output)
concat_step.add_argument(name='inputs2', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_1_output)
# concat_step.add_argument(name='inputs2', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_2_output)
big_pipeline.add_step(concat_step)
concat_step_output = concat_step.add_output('produce')

# concat_step = pipeline_module.PrimitiveStep({
#     "python_path": "d3m.primitives.dsbox.VerticalConcat",
#     "id": "dsbox-vertical-concat",
#     "version": "1.3.0",
#     "name": "DSBox vertically concat"})
# concat_step.add_argument(name='inputs', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_0_output)
# concat_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_1_output)
# concat_step.add_argument(name='inputs2', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_2_output)
# big_pipeline.add_step(concat_step)
# concat_step_output = concat_step.add_output('produce')


# unfold_step = pipeline_module.PrimitiveStep({
#     "python_path": "d3m.primitives.dsbox.Unfold",
#     "version": "1.3.0",
#     "id": "dsbox-unfold",
#     "name": "DSBox unfold"})
# unfold_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
# big_pipeline.add_step(unfold_step)
# unfold_step_output = unfold_step.add_output("produce")
# # big_pipeline.add_output(name="unfold", data_reference=big_output)

# replace_semantic_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.ReplaceSemanticTypes").metadata.query()))
# replace_semantic_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=unfold_step_output)
# big_pipeline.add_step(replace_semantic_step)
# replace_semantic_step.add_hyperparameter(name="match_logic", argument_type=pipeline_module.ArgumentType.VALUE, data="all")
# replace_semantic_step.add_hyperparameter(name="from_semantic_types", argument_type=pipeline_module.ArgumentType.VALUE, data=("https://metadata.datadrivendiscovery.org/types/PredictedTarget","https://metadata.datadrivendiscovery.org/types/Target"))
# replace_semantic_step.add_hyperparameter(name="to_semantic_types", argument_type=pipeline_module.ArgumentType.VALUE, data=("https://metadata.datadrivendiscovery.org/types/CategoricalData","https://metadata.datadrivendiscovery.org/types/Attribute"))
# replace_semantic_step_output = replace_semantic_step.add_output("produce")


encode_res_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Encoder").metadata.query()))
encode_res_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
big_pipeline.add_step(encode_res_step)
encode_res_step_output = encode_res_step.add_output("produce")

concat_step1 = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.HorizontalConcat").metadata.query()))
concat_step1.add_argument(name="left", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=encode_res_step_output)
concat_step1.add_argument(name="right", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=pipeline_output)
concat_step1.add_hyperparameter(name="use_index", argument_type=pipeline_module.ArgumentType.VALUE, data=False)
big_pipeline.add_step(concat_step1)
concat_output1 = concat_step1.add_output("produce")




model_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.sklearn_wrap.SKBernoulliNB").metadata.query()))
model_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_output1)
model_step.add_argument(name="outputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=target)
big_pipeline.add_step(model_step)
big_output = model_step.add_output("produce")
final_output = big_pipeline.add_output(name="final", data_reference=big_output)






# concat_step0 = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.HorizontalConcat").metadata.query()))
# concat_step0.add_argument(name="left", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_0_output)
# concat_step0.add_argument(name="right", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_1_output)
# concat_step0.add_hyperparameter(name="use_index", argument_type=pipeline_module.ArgumentType.VALUE, data=False)
# big_pipeline.add_step(concat_step0)
# concat_output0 = concat_step0.add_output("produce")


# concat_step1 = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.HorizontalConcat").metadata.query()))
# concat_step1.add_argument(name="left", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_output0)
# concat_step1.add_argument(name="right", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_2_output)
# concat_step1.add_hyperparameter(name="use_index", argument_type=pipeline_module.ArgumentType.VALUE, data=False)
# big_pipeline.add_step(concat_step1)
# concat_output1 = concat_step1.add_output("produce")


# concat_step2 = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.HorizontalConcat").metadata.query()))
# concat_step2.add_argument(name="left", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_output1)
# concat_step2.add_argument(name="right", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=pipeline_output)
# concat_step2.add_hyperparameter(name="use_index", argument_type=pipeline_module.ArgumentType.VALUE, data=False)
# big_pipeline.add_step(concat_step2)
# concat_output2 = concat_step2.add_output("produce")

# # testing: Extract target_columns, Randomforest
# replace_semantic_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.ReplaceSemanticTypes").metadata.query()))
# replace_semantic_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_output2)
# big_pipeline.add_step(replace_semantic_step)
# replace_semantic_step.add_hyperparameter(name="match_logic", argument_type=pipeline_module.ArgumentType.VALUE, data="all")
# replace_semantic_step.add_hyperparameter(name="from_semantic_types", argument_type=pipeline_module.ArgumentType.VALUE, data=("https://metadata.datadrivendiscovery.org/types/PredictedTarget","https://metadata.datadrivendiscovery.org/types/Target"))
# replace_semantic_step.add_hyperparameter(name="to_semantic_types", argument_type=pipeline_module.ArgumentType.VALUE, data=("https://metadata.datadrivendiscovery.org/types/CategoricalData","https://metadata.datadrivendiscovery.org/types/Attribute"))
# replace_semantic_step_output = replace_semantic_step.add_output("produce")


# encode_res_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Encoder").metadata.query()))
# encode_res_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=replace_semantic_step_output)
# big_pipeline.add_step(encode_res_step)
# encode_res_step_output = encode_res_step.add_output("produce")


# model_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.sklearn_wrap.SKBernoulliNB").metadata.query()))
# model_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=encode_res_step_output)
# model_step.add_argument(name="outputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=target)
# big_pipeline.add_step(model_step)
# big_output = model_step.add_output("produce")
# big_pipeline.add_output(name="final", data_reference=big_output)

# runtime = runtime_module.Runtime(big_pipeline)



# concat_step = pipeline_module.PrimitiveStep({
#     "python_path": "d3m.primitives.dsbox.VerticalConcat",
#     "id": "dsbox-vertical-concat",
#     "version": "1.3.0",
#     "name": "DSBox vertically concat"})
# concat_step.add_argument(name='inputs', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_0_output)
# concat_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_1_output)
# concat_step.add_argument(name='inputs2', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_2_output)
# big_pipeline.add_step(concat_step)
# concat_step_output = concat_step.add_output('produce')


# unfold_step = pipeline_module.PrimitiveStep({
#     "python_path": "d3m.primitives.dsbox.Unfold",
#     "version": "1.3.0",
#     "id": "dsbox-unfold",
#     "name": "DSBox unfold"})
# unfold_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
# big_pipeline.add_step(unfold_step)
# big_output = unfold_step.add_output("produce")
# big_pipeline.add_output(name="unfold", data_reference=big_output)
dataset = container.Dataset.load('file:///Users/muxin/Desktop/ISI/dsbox-env/data/datasets/seed_datasets_current/38_sick/38_sick_dataset/datasetDoc.json')
set_target_column(dataset)


# # import logging
# # _logger = logging.getLogger(__name__)

test_fitted = FittedPipeline(pipeline=big_pipeline, dataset_id="aaa", log_dir="log")
test_fitted.runtime.set_not_use_cache()
test_fitted.fit(inputs=[dataset])
test_fitted.produce(inputs=[dataset])

# import pdb
# pdb.set_trace()
# test_fitted.save("./")
# my_pip = FittedPipeline.load(folder_loc="./", pipeline_id="08c69b23-7f63-48a0-8ebe-e90844e8df5a", log_dir="rua")
# my_pip.runtime.set_not_use_cache()

# my_pip.produce(inputs=[dataset])
# import pdb
# pdb.set_trace()
# my_pip.produce()

print("???")
