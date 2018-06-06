import datetime
import typing
import uuid

import dateparser  # type: ignore
import jsonpath_ng  # type: ignore

from networkx import nx  # type: ignore

from d3m import exceptions, utils
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.pipeline import Pipeline, PipelineStep, StepBase, PrimitiveStep, PlaceholderStep, SubpipelineStep, ArgumentType, PlaceholderStep, Resolver, PIPELINE_SCHEMA_VALIDATOR
from d3m.primitive_interfaces.base import PrimitiveBaseMeta

# Define separate extended pipe step enum, because Python cannot extend Enum
class ExtendedPipelineStep(utils.Enum):
    TEMPLATE = 1

class SemanticType(utils.Enum):
    """
    Semantic type of template nodes.
    """
    # Need to map this to primitives in dsbox.planner.common.ontology.D3mOntology
    UNDEFINED = 1
    ENCODER = 2
    IMPUTER = 3
    FEATURER_GENERATOR = 4
    FEATURER_SELECTOR = 5
    CLASSIFIER = 6
    REGRESSOR = 7

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

    def __init__(self, name: str, semantic_type: SemanticType, resolver: Resolver = None) -> None:
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
        step = PrimitiveStep(metadata, resolver = resolver)

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
                        'type' : argument_type,
                        'data' : value
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
                print(binding[template_step.name].hyperparams)
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
    names =[]
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

