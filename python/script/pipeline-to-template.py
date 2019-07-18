import argparse
import pprint

from collections import OrderedDict

from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType

pipeline_file = '/user_opt/dsbox/primitives/v2019.6.7/Distil/d3m.primitives.data_transformation.encoder.DistilTextEncoder/0.1.0/pipelines/0ed6fbca-2afd-4ba6-87cd-a3234e9846c3.json'

def generate_template(pipeline_file: str) -> dict:
    with open(pipeline_file) as f:
        pipeline = Pipeline.from_json(f)

    steps = []
    for i, step in enumerate(pipeline.steps):
        if not isinstance(step, PrimitiveStep):
            raise ValueError('Can only handle PrimitiveSteps')
        step_name = f'steps.{i}'
        hyperparameters = {}
        for name, value in step.hyperparams.items():
            if value['type'] == ArgumentType.VALUE:
                hyperparameters[name] = value['data']
            else:
                raise ValueError(f'Do not know how to parse hyperparam: {str(value)}')
        arguments = []
        argument_keys = set(step.arguments.keys())
        for argument_name in ['inputs', 'outputs', 'reference']:
            if argument_name in argument_keys:
                argument_keys.remove(argument_name)
                if step.arguments[argument_name]['type'] == ArgumentType.CONTAINER:
                    if step.arguments[argument_name]['data'] == 'inputs.0':
                        arguments.append('template_input')
                    elif step.arguments[argument_name]['data'].startswith('steps.') and step.arguments['inputs']['data'].endswith('.produce'):
                        arguments.append(step.arguments[argument_name]['data'][:-8])
                    else:
                        raise ValueError(f"Do not know how to parse argument: {step.arguments['inputs']['data']}")
                else:
                    raise ValueError(f"Do not know how to parse argument type: {step.arguments['inputs']['type']}")
        if len(argument_keys) > 0:
            for argument_name in argument_keys:
                print(argument_name, step.arguments[argument_name])
            raise ValueError(f"Unused arguments: {argument_keys}")
        primitive = OrderedDict()
        primitive['primitive'] = str(step.primitive)
        primitive['hyperparameters'] = hyperparameters
        step = OrderedDict()
        step['name'] = step_name
        step['primitives'] = [primitive]
        step['inputs'] = arguments
        steps.append(step)
    template = OrderedDict()
    template['name'] = pipeline.id if pipeline.name is None else pipeline.name
    template['taskType'] = {'TaskType'}
    template['taskSubtype'] = {'TaskSubtype'}
    template['inputType'] = {'table'}
    template['output'] = step_name
    template['steps'] = steps
    return template

def print_value(value):
    if isinstance(value, str):
        return (f"'{value}'")
    else:
        return (str(value))


def print_template(template: dict, level=0, indent=4, trailing=''):
    if isinstance(template, OrderedDict):
        print(' '*(level*indent-1),'{')
        level = level + 1
        for key, value in template.items():
            if isinstance(value, list) and isinstance(value[0], dict):
                print(' '*(level*indent-1), f"'{key}': [")
                for element in value:
                    print_template(element, level=level+1, trailing=',')
                print(' '*(level*indent-1),'],')
            elif key=='hyperparameters':
                print(' '*(level*indent-1), f"'{key}': "+"{")
                for hyper_name, hyper_value in value.items():
                    print(' '*((level+1)*indent-1), f"'{hyper_name}': [{print_value(hyper_value)}],")
                print(' '*(level*indent-1),'},')

            else:
                print(' '*(level*indent-1), f"'{key}': {print_value(value)},")
        level = level - 1
        print(' '*(level*indent-1),'}'+trailing)
    else:
        print('SHOULD NOT BE HERE')
        print(type(template))
        print(template)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert pipeline json to template",
    )
    parser.add_argument('pipeline_file', help="Pipeline json file")

    args = parser.parse_args()

    pipeline_file = args.pipeline_file

    template = generate_template(pipeline_file)
    print_template(template)
