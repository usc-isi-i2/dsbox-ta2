import datetime
import typing
import uuid
import numpy as np
import dateparser  # type: ignore
import jsonpath_ng  # type: ignore
import pprint
from networkx import nx  # type: ignore
import copy

from d3m import exceptions, utils, index as d3m_index
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.pipeline import Pipeline, PipelineStep, StepBase, \
    PrimitiveStep, PlaceholderStep, SubpipelineStep, ArgumentType, \
    PlaceholderStep, Resolver, PIPELINE_SCHEMA_VALIDATOR, PipelineContext
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
        self.primitive = d3m_index.search()
        self.argmentsmapper = {
            "container": ArgumentType.CONTAINER,
            "data": ArgumentType.DATA,
            "value": ArgumentType.VALUE,
            "primitive": ArgumentType.PRIMITIVE
        }
        self.stepcheck = None  # Generate a step check matrix

        self.step_number = {}
        self.addstep_mapper = {
            ("<class 'd3m.container.pandas.DataFrame'>", "<class 'd3m.container.numpy.ndarray'>"): "d3m.primitives.data.DataFrameToNDArray",
            # ("<class 'd3m.container.pandas.DataFrame'>", "<class 'd3m.container.numpy.ndarray'>"): "d3m.primitives.sklearn_wrap.SKImputer",
            ("<class 'd3m.container.numpy.ndarray'>", "<class 'd3m.container.pandas.DataFrame'>"): "d3m.primitives.data.NDArrayToDataFrame"
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

        Examples:
            configuration_point =
            {
                "my_step1" : {
                    "primitive": "dsbox.a.b",
                    "hyperparameters": {
                        "x": 1
                    }
                },
                "my_step2" : {
                    "primitive": "sklearn.a.b",
                    "hyperparameters": {}
                }
            }
            dstemp = DSBoxTemplate(...)
            dstemp.to_pipeline(configuration_point)
        """
        # print("*" * 20)
        # print("[INFO] to_pipeline:")
        # pprint(configuration_point)
        # return self._to_pipeline(configuration_point)

        # add inputs to the configuration point
        ioconf = self.add_inputs_to_confPonit(configuration_point)

        # binding = configuration_point
        binding, sequence = self.add_intermediate_type_casting(ioconf)
        # print("[INFO] Binding:")
        # pprint(binding)
        return self._to_pipeline(binding, sequence)

    def add_inputs_to_confPonit(self, configuration_point: ConfigurationPoint) -> ConfigurationPoint:

        io_conf = copy.deepcopy(configuration_point)
        for step in self.template['steps']:
            io_conf[step['name']]['inputs'] = step['inputs']
        return io_conf

    def add_intermediate_type_casting(
            self, configuration_point: ConfigurationPoint) \
            -> ConfigurationPoint:
        """
        This method parses the information in the template and adds the
        necessary type casting primitives in the pipeline. These type
        information is associated with each individual primitive present in
        the template and is governed by d3m's primitive rules.
        Args:
            configuration_point: Configuration

        Returns:
            binding: Configuration

        """
        # binding = ....
        binding = configuration_point
        checked_binding = {}
        sequence = []
        # for step in self.template["steps"]:
        for step_num, step in enumerate(self.template["steps"]):
            # First element in the inputs array is always the input of the
            # step in configuration point. In order to check the need for
            # adding intermediate step we first extract metadata information
            # of steps and by comparing the IO type information we decide on
            # whether intermediate type caster is necessary or not

            inputs = step["inputs"]
            fill_in = copy.deepcopy(inputs)
            name = step["name"]
            for in_arg in inputs:
                in_primitive_value = \
                    d3m_index.get_primitive(binding[name]["primitive"]).metadata.query()[
                        "primitive_code"]["class_type_arguments"]["Inputs"]
                if in_arg == "template_input":
                    continue

                # Check if the input name is valid and available in template
                if in_arg not in binding:
                    print("[ERROR] step {} input {} is not available!".format(step_num, in_arg))
                    print("binding: ")
                    pprint(binding)
                    return 1

                # get information of the producer of the input
                out_primitive_value = \
                    d3m_index.get_primitive(binding[in_arg]["primitive"]).metadata.query()[
                        "primitive_code"]["class_type_arguments"]["Outputs"]
                if not self.iocompare(in_primitive_value,
                                      out_primitive_value):
                    check_key = (str(out_primitive_value),
                                 str(in_primitive_value))
                    print("[INFO] Different types!")
                    try:
                        # inter_name = "{}_{}_{}".format(name,in_arg,solution)
                        solution = self.addstep_mapper[check_key]
                        inter_name = "{}_{}_{}".format(name, in_arg, solution)
                        intermediate_step = {
                            "primitive": solution,
                            "hyperparameters": {},
                            "inputs": [in_arg]
                        }
                        # binding[inter_name] = intermediate_step
                        # binding[name]['inputs'][0] = inter_name
                        # checked_binding[inter_name] = intermediate_step
                        pos = binding[name]["inputs"].index(in_arg)
                        # checked_binding[name]["inputs"][pos] = inter_name
                        checked_binding[inter_name] = intermediate_step
                        fill_in[pos] = in_arg
                        sequence.append(inter_name)
                        print("[INFO] ", solution, "added to step",
                              name)
                    except:
                        print("Warning!", name,
                              "'s primitive",
                              # Fixme:
                              # conf_step[-1]["primitive"],
                              "'s inputs does not match",
                              binding[in_arg][-1]["primitive"],
                              "and there is no converter found")
            mystep = {
                "primitive": binding[name]["primitive"],
                "hyperparameters": binding[name]["hyperparameters"],
                "inputs": fill_in
            }
            if "runtime" in step:
                mystep["runtime"] = step["runtime"]

            sequence.append(name)
            checked_binding[name] = mystep

        return checked_binding, sequence

    def iocompare(self, i, o):
        try:
            i = i.__args__
            if o in i:
                return True
        except:
            if o == i:
                return True
        return False

    def bind_primitive_IO(self, primitive: PrimitiveStep, *templateIO):
        #print(templateIO)
        if len(templateIO) > 0:
            primitive.add_argument(
                name="inputs",
                argument_type=ArgumentType.CONTAINER,
                data_reference=templateIO[0])

        if len(templateIO) > 1:
            arguments = primitive.primitive.metadata.query()['primitive_code']['instance_methods']['set_training_data']['arguments']
            if "outputs" in arguments:
                # Some primitives (e.g. GreedyImputer) require "outputs", while others do
                # not (e.g. MeanImputer)
                primitive.add_argument("outputs", ArgumentType.CONTAINER,
                                       templateIO[1])
        if len(templateIO) > 2:
            raise exceptions.InvalidArgumentValueError(
                "Should be less than 3 arguments!")

    def _to_pipeline(self, binding, sequence) -> Pipeline:
        """
        Args:
            binding:

        Returns:

        """

        # define an empty pipeline with the general dataset input primitive
        # generate empty pipeline with i/o/s/u =[]
        # pprint(binding)
        # print(sequence)
        # print("[INFO] list:",list(map(str, PipelineContext)))
        pipeline = Pipeline(name="dsbox_" + str(id(binding)),
                            context=PipelineContext.PRETRAINING) #'PRETRAINING'
        templateinput = pipeline.add_input("input dataset")

        # save temporary output for another step to take as input
        outputs = {}
        outputs["template_input"] = templateinput

        # iterate through steps in the given binding and add each step to the
        #  pipeline. The IO and hyperparameter are also handled here.
        for i, step in enumerate(sequence):
            self.step_number[step] = i
            #primitive_step = PrimitiveStep(self.primitive[binding[step]["primitive"]].metadata.query())
            primitive_name = binding[step]["primitive"]
            if primitive_name in self.primitive:
                primitive_desc = dict(d3m_index.get_primitive(primitive_name).metadata.query())

                # Add information Runtime
                if "runtime" in binding[step]:
                    primitive_desc["runtime"] = binding[step]["runtime"]
                primitive_step = PrimitiveStep(primitive_desc)
            else:
                raise exceptions.InvalidArgumentValueError("Error, can't find the primitive : ", primitive_name)

            if binding[step]["hyperparameters"] != {}:
                hyper = binding[step]["hyperparameters"]
                for hyperName in hyper:
                    primitive_step.add_hyperparameter(
                        # argument_type should be fixed type not the type of the data!!
                        name=hyperName, argument_type=self.argmentsmapper["value"],
                        data=hyper[hyperName])
            templateIO = binding[step]["inputs"]

            # first we need to extract the types of the primtive's input and
            # the generators's output type.
            # then we need to compare those and in case we have different
            # types, add the intermediate type caster in the pipeline
            # print(outputs)
            self.bind_primitive_IO(primitive_step,
                                   *map(lambda io: outputs[io], templateIO))
            pipeline.add_step(primitive_step)
            outputs[step] = primitive_step.add_output("produce")
        # END FOR

        # Add final output as the prediction of target attribute
        general_output = outputs[self.template["steps"][-1]["name"]]
        # print(general_output)
        pipeline.add_output(general_output, "predictions of input dataset")

        return pipeline

    def generate_configuration_space(self) -> SimpleConfigurationSpace:
        steps = self.template["steps"]
        conf_space = {}
        for each_step in steps:
            name = each_step["name"]
            values = []

            # description: typing.Dict
            for description in each_step["primitives"]:
                value_step = []
                # primitive with no hyperparameters
                if isinstance(description, str):
                    value_step.append({
                        "primitive": description,
                        "hyperparameters": {}
                    })
                # one primitive with hyperparamters
                elif isinstance(description, dict):
                    value_step += self.description_to_configuration(description)
                # list of primitives
                elif isinstance(description, list):
                    for prim in description:
                        value_step += self.description_to_configuration(prim)
                else:
                    # other data format, not supported, raise error
                    print("Error: Wrong format of the description: "
                          "Unsupported data format found : ",type(description))

                values += value_step

            # END FOR
            if len(values) > 0:
                conf_space[name] = values
        # END FOR
        return SimpleConfigurationSpace(conf_space)

    def description_to_configuration(self, description):
        value = []
        # if the desciption is an dictionary:
        # it maybe a primitive with hyperparameters
        if "primitive" not in description:
            print("Error: Wrong format of the configuration space data: "
                  "No primitive name found!")
        else:
            if "hyperparameters" not in description:
                description["hyperparameters"] = {}

            # go through the hypers and if anyone has empty value just remove it
            hyperDict = dict(filter(lambda kv: len(kv[1]) > 0,
                                    description["hyperparameters"].items()))

            # go through the hyper values for single tuples and convert them
            # to a list with single tuple element
            hyperDict = dict(map(
                lambda kv:
                (kv[0], [kv[1]]) if isinstance(kv[1],tuple) else (kv[0], kv[1]),
                hyperDict.items()
            ))

            # iterate through all combinations of the hyperparamters and add
            # each as a separate configuration point to the space
            for hyper in _product_dict(hyperDict):
                value.append({
                    "primitive": description["primitive"],
                    "hyperparameters": hyper,
                })
        return value

    def get_target_step_number(self):
        #self.template[0].template['output']
        return self.step_number[self.template['output']]

    def get_output_step_number(self):
        return self.step_number[self.template['output']]


def _product_dict(dct):
    keys = dct.keys()
    vals = dct.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))
