import datetime
import typing
import uuid
import numpy as np
import dateparser  # type: ignore
import jsonpath_ng  # type: ignore
import pprint
from networkx import nx  # type: ignore

from d3m import exceptions, utils, index
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.pipeline import Pipeline, PipelineStep, StepBase, PrimitiveStep, PlaceholderStep, SubpipelineStep, ArgumentType, PlaceholderStep, Resolver, PIPELINE_SCHEMA_VALIDATOR
from d3m.primitive_interfaces.base import PrimitiveBaseMeta
from .configuration_space import DimensionName, ConfigurationSpace, SimpleConfigurationSpace, ConfigurationPoint
# from dsbox.template.search import TemplateDimensionalRandomHyperparameterSearch, TemplateDimensionalSearch, ConfigurationSpace, SimpleConfigurationSpace, PythonPath, DimensionName
# Define separate extended pipe step enum, because Python cannot extend Enum

from itertools import zip_longest, product
from pprint import pprint


class ExtendedPipelineStep(utils.Enum):
    TEMPLATE = 1


HYPERPARAMETER_DIRECTIVE: str = 'dsbox_hyperparameter_directive'


class HyperparamDirective(utils.Enum):
    """
    Specify how to choose hyperparameters
    """
    DEFAULT = 1
    RANDOM = 2


TS = typing.TypeVar('TS', bound='TemplateStep')


class TemplateStep(PlaceholderStep):
    """
    Class representing a template step in the pipeline.

    Attributes
    ----------
    name : str
        An unique user friendly name for this node
    semantic_type : SemanticType
        Type of the primitives that should used to fill in the template
    expected_arguments : Dict[str, Dict]
        Arguments the template step is expecting

    Parameters
    ----------
    semantic_type : SemanticType
        Type of the primitives that should used to fill in the template
    expected_arguments : Dict[str, Dict]
        Arguments the template step is expecting
    """

    def __init__(self, name: str, semantic_type: str, resolver: Resolver = None) -> None:
        super().__init__(resolver=resolver)
        self.name = name
        self.semantic_type = semantic_type
        self.expected_arguments: typing.Dict[str, typing.Dict] = {}

    def add_expected_argument(self, name: str, argument_type: typing.Any):

        if name in self.expected_arguments:
            raise exceptions.InvalidArgumentValueError("Argument with name '{name}' already exists.".format(name=name))

        if argument_type not in [ArgumentType.CONTAINER, ArgumentType.DATA]:
            raise exceptions.InvalidArgumentValueError("Invalid argument type: {argument_type}".format(argument_type=argument_type))

        self.expected_arguments[name] = {
            'type': argument_type
        }

    @classmethod
    def from_json(cls: typing.Type[TS], step_description: typing.Dict, *, resolver: Resolver = None) -> TS:
        step = cls(step_description['name'], step_description['semanticType'], resolver=resolver)

        for input_description in step_description['inputs']:
            step.add_input(input_description['data'])

        for output_description in step_description['outputs']:
            step.add_output(output_description['id'])

        return step

    def to_json(self) -> typing.Dict:
        step_description = {
            'type': PipelineStep.PLACEHOLDER,
            'subtype': ExtendedPipelineStep.TEMPLATE,
            'inputs': [self._input_to_json(data_reference) for data_reference in self.inputs],
            'outputs': [self._output_to_json(output_id) for output_id in self.outputs],
            'name': self.name,
            'semanticType': self.semantic_type
        }

        return step_description


TP = typing.TypeVar('TP', bound='TemplatePipeline')


class TemplatePipeline(Pipeline):
    """
    Pipeline with template steps
    """

    def __init__(self, pipeline_id: str = None, *, context: typing.Any, created: datetime.datetime = None,
                 source: typing.Dict = None, name: str = None, description: str = None) -> None:
        super().__init__(pipeline_id, context=context, created=created, source=source, name=name, description=description)
        self.template_nodes: typing.Dict[str, TemplateStep] = {}

    def add_step(self, step: StepBase) -> None:
        super().add_step(step)

        if isinstance(step, TemplateStep):
            if step.name in self.template_nodes:
                raise exceptions.InvalidArgumentValueError("TemplateStep '{}' already in pipeline".format(step.name))
            self.template_nodes[step.name] = step

    def to_step(cls, metadata: dict, resolver: Resolver = None) -> PrimitiveStep:
        """
        Convenience method for generating PrimitiveStep from primitive id
        """
        step = PrimitiveStep(metadata, resolver=resolver)

        # Set hyperparameters
        if HYPERPARAMETER_DIRECTIVE in metadata:
            directive = metadata[HYPERPARAMETER_DIRECTIVE]
            if directive == HyperparamDirective.RANDOM:
                primitive = resolver.get_primitive(metadata)
                hyperparams_class = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                hyperparams = {}
                for key, value in hyperparams_class.sample().items():
                    if value is None or issubclass(type(value), int) or issubclass(type(value), float) or issubclass(type(value), str):
                        argument_type = ArgumentType.DATA
                    else:
                        raise ValueError('TemplatePipeline.to_step(): Need to add case for type: {}'.format(type(value)))
                    hyperparams[key] = {
                        'type': argument_type,
                        'data': value
                    }
                step.hyperparams = hyperparams

        return step

    def to_steps(cls, primitive_map: typing.Dict[str, dict], resolver: Resolver = None) -> typing.Dict[str, PrimitiveStep]:
        """
        Convenience method for generating PrimitiveStep from primitive id
        """
        result = {}
        for template_node_name, metadata in primitive_map.items():
            result[template_node_name] = cls.to_step(metadata, resolver)
        return result

    def get_pipeline(self, binding: typing.Dict[str, PrimitiveStep] = {}, pipeline_id: str = None, *, context: typing.Any,
                     created: datetime.datetime = None, source: typing.Dict = None, name: str = None, description: str = None) -> Pipeline:
        """
        Generates regular Pipeline from this pipeline template.

        Parameters
        ----------
        binding : typing.Dict[str, PrimitiveStep]
            Mapping from template node name to Step
        pipeline_id : str
            Optional ID for the pipeline. If not provided, it will be automatically generated.
        context : PipelineContext
            In which context was the pipeline made.
        created : datetime
            Optional timestamp of pipeline creation in UTC timezone. If not provided, the current time will be used.
        source : Dict
            Description of source. Optional.
        name : str
            Name of the pipeline. Optional.
        description : str
            Description of the pipeline. Optional.

        """
        if not set(self.template_nodes.keys()) <= set(binding.keys()):
            raise exceptions.InvalidArgumentValueError("Not all template steps have binding: {}".format(self.template_nodes.keys()))

        if source is None:
            source = {}
        source['from_template'] = self.id

        result = Pipeline(pipeline_id, context=context, created=created, source=source, name=name, description=description)

        for i, template_step in enumerate(self.steps):
            if isinstance(template_step, TemplateStep):
                print('Hyperparam binding for template step {} ({}) : {}'.format(
                    template_step.name, template_step.semantic_type, binding[template_step.name].hyperparams))
                result.add_step(binding[template_step.name])
            elif isinstance(template_step, PrimitiveStep):
                result.add_step(PrimitiveStep(template_step.primitive_description))
            elif isinstance(template_step, SubpipelineStep):
                result.add_step(SubpipelineStep(template_step.pipeline_id, resolver=template_step.resolver))
            elif isinstance(template_step, PlaceholderStep):
                result.add_step(PlaceholderStep(template_step.resolver))
            else:
                raise exceptions.InvalidArgumentValueError("Unkown step type: {}".format(type(template_step)))

        for template_step, step in zip(self.steps, result.steps):
            # add ouptuts
            for output in template_step.outputs:
                step.add_output(output)

            # add arguments or inputs
            if isinstance(template_step, PrimitiveStep):
                for name, detail in template_step.arguments.items():
                    step.add_argument(name, detail['type'], detail['data'])
            elif isinstance(template_step, TemplateStep):
                for ((name, detail), data) in zip(template_step.expected_arguments.items(), template_step.inputs):
                    step.add_argument(name, detail['type'], data)
            else:
                for input in template_step.inputs:
                    step.add_input(input)

        for input in self.inputs:
            result.add_input(input['name'])

        for output in self.outputs:
            result.add_output(output['data'], output['name'])
        return result

    @classmethod
    def from_json(cls: typing.Type[TP], pipeline_description: typing.Dict, *, resolver: Resolver = None) -> TP:
        PIPELINE_SCHEMA_VALIDATOR.validate(pipeline_description)

        # If no timezone information is provided, we assume UTC. If there is timezone information,
        # we convert timestamp to UTC
        created = dateparser.parse(pipeline_description['created'], settings={'TIMEZONE': 'UTC'})
        context = cls._get_context(pipeline_description)
        source = cls._get_source(pipeline_description)

        pipeline = cls(
            pipeline_id=pipeline_description['id'], created=created, context=context, source=source,
            name=pipeline_description.get('name', None), description=pipeline_description.get('description', None)
        )

        for input_description in pipeline_description['inputs']:
            pipeline.add_input(input_description.get('name', None))

        for step_description in pipeline_description['steps']:
            if step_description['type'] == PipelineStep.PLACEHOLDER and 'subtype' in step_description:
                step = cls._get_step_class(step_description['subtype']).from_json(step_description, resolver=resolver)
            else:
                step = cls._get_step_class(step_description['type']).from_json(step_description, resolver=resolver)
            pipeline.add_step(step)

        for output_description in pipeline_description['outputs']:
            pipeline.add_output(output_description['data'], output_description.get('name', None))

        for user_description in pipeline_description.get('users', []):
            pipeline.add_user(user_description)

        pipeline.check()

        return pipeline

    @classmethod
    def _get_step_class(cls, step_type: typing.Any) -> StepBase:
        if step_type == ExtendedPipelineStep.TEMPLATE:
            return TemplateStep
        else:
            return super()._get_step_class(step_type)


def to_digraph(pipeline: Pipeline) -> nx.DiGraph:
    """
    Convert pipeline to directed graph.
    """
    graph = nx.DiGraph()
    names = []
    for step in pipeline.steps:
        if isinstance(step, PrimitiveStep):
            names.append(str(step.primitive).split('.')[-1])
        elif isinstance(step, SubpipelineStep):
            names.append(step.pipeline_id)
        elif isinstance(step, TemplateStep):
            names.append('<{}>'.format(step.name))
        elif isinstance(step, PlaceholderStep):
            names.append('<{}>'.format(PlaceholderStep))
        else:
            names.append('UNKNOWN')

    for i, step in enumerate(pipeline.steps):
        if isinstance(step, PrimitiveStep):
            links = [arg['data'] for arg in step.arguments.values()]
        else:
            links = step.inputs  # type: ignore
        for link in links:
            origin = link.split('.')[0]
            source = link.split('.')[1]
        if origin == 'steps':
            graph.add_edge(names[int(source)], names[i])
        else:
            graph.add_edge(origin, names[i])
    return graph


class DSBoxTemplate():
    def __init__(self):
        self.primitive = index.search()
        self.argmentsmapper = {
            "container": ArgumentType.CONTAINER,
            "data": ArgumentType.DATA,
            "value": ArgumentType.VALUE,
            "primitive": ArgumentType.PRIMITIVE
        }
        self.stepcheck = None  # Generate a step check matrix

        self.step_number = {}
        self.addstep_mapper = {
            ("<class 'd3m.container.pandas.DataFrame'>", "<class 'd3m.container.numpy.ndarray'>"): "d3m.primitives.datasets.DataFrameToNDArray",
            # ("<class 'd3m.container.pandas.DataFrame'>", "<class 'd3m.container.numpy.ndarray'>"): "d3m.primitives.sklearn_wrap.SKImputer",
            ("<class 'd3m.container.numpy.ndarray'>", "<class 'd3m.container.pandas.DataFrame'>"): "d3m.primitives.datasets.NDArrayToDataFrame"
        }

        # Need to be set by subclass inheriting DSBoxTemplate
        # self.template = ""

    def add_stepcheck(self):
        check = np.zeros(shape=(len(self.primitive), len(self.primitive))).astype(int)
        for i, v in enumerate(self.primitive.keys()):
            inputs = self.primitive[v].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"]
            for j, u in enumerate(self.primitive.keys()):
                outputs = self.primitive[u].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"]
                try:
                    inp = inputs.__args__
                    if outputs in inp:
                        check[i][j] = 1
                except:
                    if inputs == outputs:
                        check[i][j] = 1
        self.stepcheck = check

    def to_pipeline(self, configuration_point: ConfigurationPoint) -> Pipeline:
        """
        converts the configuration point to the executable pipeline based on
        ta2 competitions format
        Args:
            configuration_point (ConfigurationPoint):

        Returns:
            The executable pipeline with full hyperparameter settings
        """
        # print("[INFO] to_pipeline:")
        # pprint(configuration_point)
        # return self._to_pipeline(configuration_point)

        # configuration_point =
        # {
        #     "my_step1" : {
        #         "primitive": "dsbox.a.b",
        #         "hyperparameters": {
        #             "x": 1
        #         }
        #     },
        #     "my_step2" : {
        #         "primitive": "sklearn.a.b",
        #         "hyperparameters": {}
        #     }
        # }

        # do reasoning
        # binding = ....
        binding = {}
        for step in self.template["steps"]:
            sub_steps = []
            if len(step['primitives']) == 1:
                if isinstance(step['primitives'][0], str):
                    sub_step = {
                        'primitive': step['primitives'][0],
                        'hyperparameters': {}
                    }
                else:
                    # is dict
                    print(step, "is dict")
                    sub_step = {
                        'primitive': step['primitives'][0]['primitive'],
                        # 'hyperparameters': step['primitives'][0]['hyperparameters']
                        "hyperparameters": {}
                    }

                sub_steps.append(sub_step)
            else:
                sub_step = configuration_point[step["name"]]
                sub_steps.append(sub_step)
            # print("sub_step", sub_step)
            inputs = step["inputs"]
            # print(step["name"], "+++++", inputs)
            if len(inputs) == 1:  # one intputs
                if inputs[0] != "template_input":
                    in_primitive_value = self.primitive[sub_steps[0]["primitive"]].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"]
                    for s in self.template["steps"]:
                        if s["name"] == inputs[0]:
                            out_primitive_value = self.primitive[binding[inputs[0]][-1]["primitive"]].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"]
                            if self.iocompare(in_primitive_value, out_primitive_value):
                                pass
                            else:
                                check_key = (str(out_primitive_value), str(in_primitive_value))
                                try:
                                    solution = self.addstep_mapper[check_key]
                                    tmp = []
                                    intermediate_step = {
                                        "primitive": solution,
                                        "hyperparameters": {}
                                    }
                                    tmp.append(intermediate_step)
                                    sub_steps.insert(0, tmp)
                                    print(solution, "added to step", step["name"])
                                except:
                                    print("Warning!", s, "'s primitive", sub_steps[0]["primitive"], "'s inputs does not match", binding[inputs[0]][-1]["primitive"], "and there is no converter found")

            else:  # severval inputs
                cnt = 0
                for i in inputs:
                    # print(sub_steps)
                    # print(sub_steps[0]["primitive"])
                    in_primitive_value = self.primitive[sub_steps[-1]["primitive"]].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"]
                    for s in self.template["steps"]:
                        if s["name"] == i:
                            out_primitive_value = self.primitive[binding[i][-1]["primitive"]].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"]
                            if self.iocompare(in_primitive_value, out_primitive_value):
                                sub_steps.insert(cnt, [])
                                cnt += 1
                            else:
                                check_key = (str(out_primitive_value), str(in_primitive_value))
                                try:
                                    solution = self.addstep_mapper[check_key]
                                    tmp = []
                                    intermediate_step = {
                                        "primitive": solution,
                                        "hyperparameters": {}
                                    }
                                    tmp.append(intermediate_step)
                                    sub_steps.insert(cnt, tmp)
                                    cnt += 1
                                    print(solution, "added to step", step["name"])
                                except:
                                    print("Warning!", step["name"], "'s primitive", sub_steps[-1]["primitive"], "'s inputs does not match", binding[i][-1]["primitive"], "and there is no converter found")

            binding[step["name"]] = sub_steps
        # print("&&&&&&&&&&&&&&&&&&&&&&&")
        # pprint.pprint(binding)
        # print("&&&&&&&&&&&&&&&&&&&&&&&")

        return self._to_pipeline(binding)

    def iocompare(self, i, o):
        try:
            i = i.__args__
            if o in i:
                return True
        except:
            if o == i:
                return True
        return False

    def _to_pipeline(self, binding) -> Pipeline:
        # binding =
        # {
        #     "my_step1": [
        #         [{
        #             "primitive": "dsbox.c.d",
        #             "hyperparameters": {
        #                 "y": 3
        #             }
        #         }],
        #         {
        #             "primitive": "dsbox.a.b",
        #             "hyperparameters": {
        #                 "x": 1
        #             }
        #         }
        #     ],
        #     "my_step2": [
        #         {
        #             "primitive": "sklearn.a.b",
        #             "hyperparameters": {}
        #         }
        #     ]
        #     "my_step3": [
        #         [{
        #             "primtive": "sklearn.ee",
        #         }],
        #         [{
        #             "primtive": "sklearn.ff",
        #             "hyperparameters": {},
        #         }],
        #
        #         {
        #             "primitive": "classifier",
        #             "hyperparameters": {}

        #         }
        #     ]
        # }

        # define an empty pipeline with the general dataset input primitive
        # generate empty pipeline with i/o/s/u =[]
        pipeline = Pipeline(name="Helloworld", context='PRETRAINING')
        templateinput = pipeline.add_input("input dataset")

        # save temporary output for another step to take as input
        outputs = {}  
        stepcount = 0

        # iterate through steps in the given binding and add each step to the
        #  pipeline. The IO and hyperparameter are also handled here.
        for index, original_step in enumerate(self.template["steps"]):
        # for index, (name, prmtv) in enumerate(binding.items()):
            name = original_step['name']
            prmtv = binding[name]

            if len(binding[name]) == 1:

                self.step_number[name] = stepcount
                primitiveStep = PrimitiveStep(self.primitive[binding[name][0]["primitive"]].metadata.query())
                pipeline.add_step(primitiveStep)
                outputs[name] = primitiveStep.add_output("produce")

                # setting the hyperparameters
                if prmtv["hyperparameters"] != {}:
                    hyper = prmtv["hyperparameters"]
                    for hyperName in hyper:
                        # TODO add support for types
                        primitiveStep.add_hyperparameter(
                            name=hyperName, argument_type=type(hyper[hyperName]),
                            data=hyper[hyperName])

                if len(original_step["inputs"]) == 1:
                    for i in original_step["inputs"]:
                        if i == "template_input":
                            primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, templateinput)
                        else:
                            primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, outputs[i])
                elif len(original_step["inputs"]) == 2:
                    primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, outputs[original_step["inputs"][0]])
                    primitiveStep.add_argument("outputs", ArgumentType.CONTAINER, outputs[original_step["inputs"][1]])
                else:
                    raise exceptions.InvalidArgumentValueError("Should be less than 3 arguments!")
                stepcount += 1
            else:
                inputs = original_step["inputs"]
                # print(inputs)

                # primitiveStep = PrimitiveStep(self.primitive[laststep["primitive"]].metadata.query())

                for pipnum, subpipeline in enumerate(binding[name][:-1]):
                    if subpipeline == []:
                        tmpname = name + ".pipeline" + str(pipnum) + ".step" + str(0)
                        outputs[tmpname] = outputs[inputs[pipnum]]

                    else:
                        for stepnum, step in enumerate(subpipeline):
                            primitiveStep = PrimitiveStep(self.primitive[step["primitive"]].metadata.query())
                            pipeline.add_step(primitiveStep)
                            tmpname = name + ".pipeline" + str(pipnum) + ".step" + str(stepnum)
                            self.step_number[tmpname] = stepcount
                            stepcount += 1
                            outputs[tmpname] = primitiveStep.add_output("produce")
                            if stepnum == 0:
                                primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, outputs[inputs[pipnum]])
                            else:
                                _tmpname = name + ".pipeline" + str(pipnum) + ".step" + str(stepnum - 1)
                                primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, outputs[_tmpname])
                            if step["hyperparameters"] != {}:
                                for key in step["hyperparameters"].keys():
                                    primitiveStep.add_hyperparameter(key, self.argmentsmapper[step["hyperparameters"][key]["type"]], step["hyperparameters"][key]["value"])
                laststep = binding[name][-1]
                primitiveStep = PrimitiveStep(self.primitive[laststep["primitive"]].metadata.query())
                pipeline.add_step(primitiveStep)
                self.step_number[name] = stepcount
                stepcount += 1
                outputs[name] = primitiveStep.add_output("produce")

                # setting the hyperparameters
                if laststep["hyperparameters"] != {}:
                    hyper = laststep["hyperparameters"]
                    for hyperName in hyper:
                        # TODO add support for types
                        primitiveStep.add_hyperparameter(
                            name=hyperName, argument_type=type(hyper[hyperName]),
                            data=hyper[hyperName])

                if len(inputs) == 1:
                    tmpkey = inputs[0] + ".pipeline0" + ".step" + str(len(inputs) - 1)
                    primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, outputs[tmpkey])
                elif len(inputs) == 2:
                    tmpkey1 = name + ".pipeline0" + ".step" + str(0)
                    tmpkey2 = name + ".pipeline1" + ".step" + str(0)
                    # print(outputs)
                    primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, outputs[tmpkey1])
                    primitiveStep.add_argument("outputs", ArgumentType.CONTAINER, outputs[tmpkey2])
                else:
                    print("cannot process more than 2 arguments")


        # END FOR

        # Add final output as the prediction of target attribute
        general_output = outputs[self.template["steps"][-1]["name"]]
        # print(general_output)
        pipeline.add_output(general_output, "predictions of input dataset")

        return pipeline

    def generate_configuration_space(self) -> SimpleConfigurationSpace:
        steps = self.template["steps"]
        conf_space = {}
        for s in steps:
            name = s["name"]
            values = []

            # description: typing.Dict
            for description in s["primitives"]:
                if isinstance(description, str):
                    value = {
                        "primitive": description,
                        "hyperparameters": {}
                    }
                else:
                    # value is a dict = {"primitive": "dsbox.a.b", "hyperparameters": {}"
                    value = {
                        "primitive": description["primitive"],
                        "hyperparameters": description["hyperparameters"],
                    }
                values.append(value)
            if len(values) > 1:
                conf_space[name] = values
        # END FOR
        return SimpleConfigurationSpace(conf_space)

    def get_target_step_number(self):
        return self.step_number[self.template['target']]

    def get_output_step_number(self):
        return self.step_number[self.template['output']]

# def _product_dict(dct):
#     keys = dct.keys()
#     vals = dct.values()
#     for instance in product(*vals):
#         yield dict(zip(keys, instance))
