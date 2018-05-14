"""
Test template code
"""

import d3m

from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType, PlaceholderStep

from importlib import reload
import dsbox.template
reload(dsbox.template)
from dsbox.template import TemplatePipeline, TemplateStep, SemanticType, ExtendedPipelineStep, to_digraph

primitive = d3m.index.search()

def generate_example():
    template = Pipeline(context='PRETRAINING')

    denormalize_step = PrimitiveStep(primitive['d3m.primitives.datasets.Denormalize'].metadata.query())
    to_DataFrame_step = PrimitiveStep(primitive['d3m.primitives.datasets.DatasetToDataFrame'].metadata.query())
    column_parser_step = PrimitiveStep(primitive['d3m.primitives.data.ColumnParser'].metadata.query())
    extract_attribute_step = PrimitiveStep(primitive['d3m.primitives.data.ExtractAttributes'].metadata.query())
    cast_1_step = PrimitiveStep(primitive['d3m.primitives.data.CastToType'].metadata.query())
    impute_step = PrimitiveStep(primitive['d3m.primitives.sklearn_wrap.SKImputer'].metadata.query())
    extract_target_step = PrimitiveStep(primitive['d3m.primitives.data.ExtractTargets'].metadata.query())
    cast_2_step = PrimitiveStep(primitive['d3m.primitives.data.CastToType'].metadata.query())
    model_step = PrimitiveStep(primitive['d3m.primitives.sklearn_wrap.SKRandomForestClassifier'].metadata.query())
    # prediction_step = PrimitiveStep(primitive['d3m.primitives.data.ConstructPredictions'].metadata.query())


    template_input = template.add_input('input dataset')

    template.add_step(denormalize_step)
    template.add_step(to_DataFrame_step)
    template.add_step(column_parser_step)
    template.add_step(extract_attribute_step)
    template.add_step(cast_1_step)
    template.add_step(impute_step)
    template.add_step(extract_target_step)
    template.add_step(cast_2_step)
    template.add_step(model_step)
    # template.add_step(prediction_step)

    denormalize_step.add_argument('inputs', ArgumentType.CONTAINER, template_input)
    denormalize_step_produce = denormalize_step.add_output('produce')

    to_DataFrame_step.add_argument('inputs', ArgumentType.CONTAINER, denormalize_step_produce)
    to_DataFrame_produce = to_DataFrame_step.add_output('produce')

    column_parser_step.add_argument('inputs', ArgumentType.CONTAINER, to_DataFrame_produce)
    column_parser_produce = column_parser_step.add_output('produce')

    extract_attribute_step.add_argument('inputs', ArgumentType.CONTAINER, column_parser_produce)
    extract_attribute_produce = extract_attribute_step.add_output('produce')

    cast_1_step.add_argument('inputs', ArgumentType.CONTAINER, extract_attribute_produce)
    cast_1_produce = cast_1_step.add_output('produce')

    impute_step.add_argument('inputs', ArgumentType.CONTAINER, cast_1_produce)
    impute_produce = impute_step.add_output('produce')

    extract_target_step.add_argument('inputs', ArgumentType.CONTAINER, column_parser_produce)
    extract_target_produce = extract_target_step.add_output('produce')

    cast_2_step.add_argument('inputs', ArgumentType.CONTAINER, extract_target_produce)
    cast_2_produce = cast_2_step.add_output('produce')


    model_step.add_argument('inputs', ArgumentType.CONTAINER, impute_produce)
    model_step.add_argument('outputs', ArgumentType.CONTAINER, cast_2_produce)
    model_produce = model_step.add_output('produce')

    # prediction_step.add_argument('inputs', column_parser_produce)
    # prediction_step.add_argument('targets', model_produce)
    # prediction_produce = prediction_step.add_output('produce')

    template_output = template.add_output(model_produce, 'predictions from the input dataset')
    return template


def generate_template():
    template = TemplatePipeline(context='PRETRAINING')

    denormalize_step = PrimitiveStep(primitive['d3m.primitives.datasets.Denormalize'].metadata.query())
    to_DataFrame_step = PrimitiveStep(primitive['d3m.primitives.datasets.DatasetToDataFrame'].metadata.query())
    column_parser_step = PrimitiveStep(primitive['d3m.primitives.data.ColumnParser'].metadata.query())
    extract_attribute_step = PrimitiveStep(primitive['d3m.primitives.data.ExtractAttributes'].metadata.query())
    cast_1_step = PrimitiveStep(primitive['d3m.primitives.data.CastToType'].metadata.query())
    impute_step = PrimitiveStep(primitive['d3m.primitives.sklearn_wrap.SKImputer'].metadata.query())
    extract_target_step = PrimitiveStep(primitive['d3m.primitives.data.ExtractTargets'].metadata.query())
    cast_2_step = PrimitiveStep(primitive['d3m.primitives.data.CastToType'].metadata.query())
    # model_step = PrimitiveStep(primitive['d3m.primitives.sklearn_wrap.SKRandomForestClassifier'].metadata.query())
    # model_step = PlaceholderStep()
    model_step = TemplateStep('modeller', SemanticType.CLASSIFIER)

    template_input = template.add_input('input dataset')

    template.add_step(denormalize_step)
    template.add_step(to_DataFrame_step)
    template.add_step(column_parser_step)
    template.add_step(extract_attribute_step)
    template.add_step(cast_1_step)
    template.add_step(impute_step)
    template.add_step(extract_target_step)
    template.add_step(cast_2_step)
    template.add_step(model_step)
    # template.add_step(prediction_step)

    denormalize_step.add_argument('inputs', ArgumentType.CONTAINER, template_input)
    denormalize_step_produce = denormalize_step.add_output('produce')

    to_DataFrame_step.add_argument('inputs', ArgumentType.CONTAINER, denormalize_step_produce)
    to_DataFrame_produce = to_DataFrame_step.add_output('produce')

    column_parser_step.add_argument('inputs', ArgumentType.CONTAINER, to_DataFrame_produce)
    column_parser_produce = column_parser_step.add_output('produce')

    extract_attribute_step.add_argument('inputs', ArgumentType.CONTAINER, column_parser_produce)
    extract_attribute_produce = extract_attribute_step.add_output('produce')

    cast_1_step.add_argument('inputs', ArgumentType.CONTAINER, extract_attribute_produce)
    cast_1_produce = cast_1_step.add_output('produce')

    impute_step.add_argument('inputs', ArgumentType.CONTAINER, cast_1_produce)
    impute_produce = impute_step.add_output('produce')

    extract_target_step.add_argument('inputs', ArgumentType.CONTAINER, column_parser_produce)
    extract_target_produce = extract_target_step.add_output('produce')

    cast_2_step.add_argument('inputs', ArgumentType.CONTAINER, extract_target_produce)
    cast_2_produce = cast_2_step.add_output('produce')


    model_step.add_expected_argument('inputs', ArgumentType.CONTAINER)
    model_step.add_expected_argument('outputs', ArgumentType.CONTAINER)
    model_step.add_input(impute_produce)
    model_step.add_input(cast_2_produce)
    model_produce = model_step.add_output('produce')

    template_output = template.add_output(model_produce, 'predictions from the input dataset')
    return template


ex = generate_example()

tp = generate_template()
values0 = {
    'modeller': PrimitiveStep(primitive['d3m.primitives.sklearn_wrap.SKRandomForestClassifier'].metadata.query())
}
tp1 = tp.get_pipeline(values0, None, context='PRETRAINING')


values = {
    'modeller': PrimitiveStep(primitive['d3m.primitives.datasets.DatasetToDataFrame'].metadata.query())
}
# Error
tp2 = tp.get_pipeline(values, None, context='PRETRAINING')



import dsbox.template.search as search
reload(search)

configuration_space = search.SimpleConfigurationSpace({
    'A' : [str(x) for x in range(10)],
    'B' : [str(x) for x in range(10)],
    'C' : [str(x) for x in range(10,20)]
})

def eval(x):
    value = 0
    for x in x.values():
        value += float(x)
    return value


df = search.DimensionalSearch(eval, configuration_space)
print()
for i in range(10):
    print(i)
    result = df.search_one_iter(max_per_dimension=3)
    print(result)

print()
for i in range(10):
    print(i)
    result = df.search_one_iter(max_per_dimension=3)
    print(result)
    result = df.search_one_iter(result[0], result[1], max_per_dimension=3)
    print(result)

######

