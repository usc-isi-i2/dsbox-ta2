import abc
import argparse
import copy
import datetime
import json
import logging
import os
import os.path
import pprint
import typing
import uuid

import dateparser  # type: ignore
import yaml  # type: ignore

from . import base as metadata_base, hyperparams
from d3m import container, exceptions, index, utils
from d3m.primitive_interfaces import base

__all__ = ('Pipeline',)

logger = logging.getLogger(__name__)

# Comma because we unpack the list of validators returned from "load_schema_validators".
PIPELINE_SCHEMA_VALIDATOR, = utils.load_schema_validators(metadata_base.SCHEMAS_PATH, metadata_base.DEFINITIONS_JSON, ('pipeline.json',))

PIPELINE_SCHEMA_VERSION = 'https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json'


# Enumeration of argument and hyper-parameter types to a primitive in a step.
ArgumentType = utils.create_enum_from_json_schema_enum(
    'ArgumentType', metadata_base.DEFINITIONS_JSON,
    'definitions[container_argument,primitive_argument,data_argument,value_argument].properties.type.oneOf[*].enum[*]',
    module=__name__,
)


PipelineContext = utils.create_enum_from_json_schema_enum(
    'PipelineContext', metadata_base.DEFINITIONS_JSON,
    'definitions.pipeline_context.oneOf[*].enum[*]',
    module=__name__,
)


PipelineStep = utils.create_enum_from_json_schema_enum(
    'PipelineStep', metadata_base.DEFINITIONS_JSON,
    'definitions.pipeline_steps.items.oneOf[*].properties.type.oneOf[*].enum[*]',
    module=__name__,
)


class TypeInfo(typing.NamedTuple):
    structural_type: type
    singleton: typing.Optional[bool]


class Resolver:
    """
    A resolver to resolve primitives and pipelines.

    It resolves primitives from available primitives on the system,
    and resolves pipelines from files in pipeline search paths.

    Attributes
    ----------
    strict_resolving : bool
        If resolved primitive does not fully match specified primitive reference, raise an exception?
    pipeline_search_paths : Sequence[str]
        A list of paths to directories with pipelines to resolve from.
        Their files should be named ``<pipeline id>.json`` or ``<pipeline id>.yml``.

    Parameters
    ----------
    strict_resolving : bool
        If resolved primitive does not fully match specified primitive reference, raise an exception?
    pipeline_search_paths : Sequence[str]
        A list of paths to directories with pipelines to resolve from.
        Their files should be named ``<pipeline id>.json`` or ``<pipeline id>.yml``.
    respect_environment_variable : bool
        Use also (colon separated) pipeline search paths from ``PIPELINES_PATH`` environment variable?
    """

    def __init__(self, *, strict_resolving: bool = False, pipeline_search_paths: typing.Sequence[str] = None, respect_environment_variable: bool = True) -> None:
        self.strict_resolving = strict_resolving

        if pipeline_search_paths is None:
            self.pipeline_search_paths: typing.List[str] = []
        else:
            self.pipeline_search_paths = typing.cast(typing.List[str], pipeline_search_paths)

        if respect_environment_variable:
            self.pipeline_search_paths += [path for path in os.environ.get('PIPELINES_PATH', '').split(':') if path]

        self._primitives_loaded = False

    def get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        primitive = self._get_primitive(primitive_description)

        # This class always resolves a primitive, or throws an exception, but subclasses might return "None".
        if primitive is not None:
            self._check_primitive(primitive_description, primitive)

        return primitive

    def get_pipeline(self, pipeline_id: str) -> 'typing.Optional[Pipeline]':
        for path in self.pipeline_search_paths:
            pipeline_path = os.path.join(path, '{pipeline_id}.json'.format(pipeline_id=pipeline_id))
            try:
                with open(pipeline_path, 'r') as pipeline_file:
                    return Pipeline.from_json(pipeline_file, resolver=self)
            except FileNotFoundError:
                pass

            pipeline_path = os.path.join(path, '{pipeline_id}.yml'.format(pipeline_id=pipeline_id))
            try:
                with open(pipeline_path, 'r') as pipeline_file:
                    return Pipeline.from_yaml(pipeline_file, resolver=self)
            except FileNotFoundError:
                pass

        raise exceptions.InvalidArgumentValueError("Unable to get pipeline '{pipeline_id}'.".format(pipeline_id=pipeline_id))

    def _get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        if not self._primitives_loaded:
            self._primitives_loaded = True

            index.load_all()

        return index.get_primitive_by_id(primitive_description['id'])

    def _check_primitive(self, primitive_description: typing.Dict, primitive: typing.Type[base.PrimitiveBase]) -> None:
        primitive_metadata = primitive.metadata.query()

        if primitive_metadata['version'] != primitive_description['version']:
            if self.strict_resolving:
                raise exceptions.MismatchError(
                    "Version for primitive '{primitive_id}' does not match the one specified in the primitive description. Primitve description version: '{primitive_version}'. Resolved primitive version: '{resolved_primitive_version}'.".format(  # noqa
                        primitive_id=primitive_metadata['id'],
                        primitive_version=primitive_description['version'],
                        resolved_primitive_version=primitive_metadata['version'],
                    )
                )
            else:
                logger.warning(
                    "Version for primitive '%(primitive_id)s' does not match the one specified in the primitive description. Primitve description version: '%(primitive_version)s'. Resolved primitive version: '%(resolved_primitive_version)s'.",  # noqa
                    {
                        'primitive_id': primitive_metadata['id'],
                        'primitive_version': primitive_description['version'],
                        'resolved_primitive_version': primitive_metadata['version'],
                    },
                )

        if primitive_metadata['python_path'] != primitive_description['python_path']:
            if self.strict_resolving:
                raise exceptions.MismatchError(
                    "Python path for primitive '{primitive_id}' does not match the one specified in the primitive description. Primitve description Python path: '{primitive_python_path}'. Resolved primitive Python path: '{resolved_primitive_python_path}'.".format(  # noqa
                        primitive_id=primitive_metadata['id'],
                        primitive_python_path=primitive_description['python_path'],
                        resolved_primitive_python_path=primitive_metadata['python_path'],
                    )
                )
            else:
                logger.warning(
                    "Python path for primitive '%(primitive_id)s' does not match the one specified in the primitive description. Primitve description Python path: '%(primitive_python_path)s'. Resolved primitive Python path: '%(resolved_primitive_python_path)s'.",  # noqa
                    {
                        'primitive_id': primitive_metadata['id'],
                        'primitive_python_path': primitive_description['python_path'],
                        'resolved_primitive_python_path': primitive_metadata['python_path'],
                    },
                )

        if primitive_metadata['name'] != primitive_description['name']:
            if self.strict_resolving:
                raise exceptions.MismatchError(
                    "Name for primitive '{primitive_id}' does not match the one specified in the primitive description. Primitve description name: '{primitive_name}'. Resolved primitive name: '{resolved_primitive_name}'.".format(  # noqa
                        primitive_id=primitive_metadata['id'],
                        primitive_name=primitive_description['name'],
                        resolved_primitive_name=primitive_metadata['name'],
                    )
                )
            else:
                logger.warning(
                    "Name for primitive '%(primitive_id)s' does not match the one specified in the primitive description. Primitve description name: '%(primitive_name)s'. Resolved primitive name: '%(resolved_primitive_name)s'.",  # noqa
                    {
                        'primitive_id': primitive_metadata['id'],
                        'primitive_name': primitive_description['name'],
                        'resolved_primitive_name': primitive_metadata['name'],
                    },
                )

        if 'digest' in primitive_description:
            assert primitive_description['digest'] is not None

            if primitive_metadata.get('digest', None) != primitive_description['digest']:
                if self.strict_resolving:
                    raise exceptions.DigestMismatchError(
                        "Digest for primitive '{primitive_id}' does not match the one specified in the primitive description. Primitve description digest: {primitive_digest}. Resolved primitive digest: {resolved_primitive_digest}.".format(  # noqa
                            primitive_id=primitive_metadata['id'],
                            primitive_digest=primitive_description['digest'],
                            resolved_primitive_digest=primitive_metadata.get('digest', None),
                        )
                    )
                else:
                    logger.warning(
                        "Digest for primitive '%(primitive_id)s' does not match the one specified in the primitive description. Primitve description digest: %(primitive_digest)s. Resolved primitive digest: %(resolved_primitive_digest)s.",  # noqa
                        {
                            'primitive_id': primitive_metadata['id'],
                            'primitive_digest': primitive_description['digest'],
                            'resolved_primitive_digest': primitive_metadata.get('digest', None),
                        },
                    )


class NoResolver(Resolver):
    """
    A resolver which never resolves anything.
    """

    def _get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        return None

    def get_pipeline(self, pipeline_id: str) -> 'typing.Optional[Pipeline]':
        return None


S = typing.TypeVar('S', bound='StepBase')


class StepBase(metaclass=utils.AbstractMetaclass):
    """
    Class representing one step in pipeline's execution.

    Attributes
    ----------
    index : int
        An index of the step among steps in the pipeline.
    resolver : Resolver
        Resolver to use.

    Parameters
    ----------
    resolver : Resolver
        Resolver to use.
    """

    def __init__(self, *, resolver: Resolver = None) -> None:
        self.resolver = self.get_resolver(resolver)

        self.index: int = None

    def get_resolver(self, resolver: typing.Optional[Resolver]) -> Resolver:
        if resolver is None:
            return Resolver()
        else:
            return resolver

    def check_add(self, existing_steps: 'typing.Sequence[StepBase]', available_data_references: typing.AbstractSet[str]) -> None:
        """
        Checks if a step can be added given existing steps and available
        data references to provide to the step. It also checks if the
        state of a step is suitable to be added at this point.

        Raises an exception if check fails.

        Parameters
        ----------
        existing_steps : Sequence[StepBase]
            Steps already in the pipeline.
        available_data_references : AbstractSet[str]
            A set of available data references.
        """

    def set_index(self, index: int) -> None:
        if self.index is not None:
            raise exceptions.InvalidArgumentValueError("Index already set to {index}.".format(index=self.index))

        self.index = index

    @abc.abstractmethod
    def get_output_data_references(self) -> typing.AbstractSet[str]:
        pass

    @classmethod
    @abc.abstractmethod
    def from_json_structure(cls: typing.Type[S], step_description: typing.Dict, *, resolver: Resolver = None) -> S:
        pass

    @abc.abstractmethod
    def to_json_structure(self) -> typing.Dict:
        pass


SP = typing.TypeVar('SP', bound='PrimitiveStep')


class PrimitiveStep(StepBase):
    """
    Class representing a primitive execution step in pipeline's execution.

    Attributes
    ----------
    primitive_description : Dict
        A description of the primitive specified for this step.
    primitive : PrimitiveBase
        A primitive class associated with this step.
    outputs : List[str]
        A list of method names providing outputs for this step.
    hyperparams : Dict[str, Dict]
        A map of of fixed hyper-parameters to their values which are set
        as part of a pipeline and should not be tuned during hyper-parameter tuning.
    arguments : Dict[str, Dict]
        A map between argument name and its description. Description contains
        a data reference of an output of a prior step (or a pipeline input).
    users : List[Dict]
        Users associated with the primitive.

    Parameters
    ----------
    primitive_description : Dict
        A description of the primitive specified for this step.
    """

    def __init__(self, primitive_description: typing.Dict, *, resolver: Resolver = None) -> None:
        super().__init__(resolver=resolver)

        self.primitive_description = primitive_description
        self.primitive = self.resolver.get_primitive(primitive_description)

        self.outputs: typing.List[str] = []
        self.hyperparams: typing.Dict[str, typing.Dict] = {}
        self.arguments: typing.Dict[str, typing.Dict] = {}
        self.users: typing.List[typing.Dict] = []

    def add_argument(self, name: str, argument_type: typing.Any, data_reference: str) -> None:
        """
        Associate a data reference to an argument of this step (and underlying primitive).

        Parameters
        ----------
        name : str
            Argument name.
        argument_type : ArgumentType
            Argument type.
        data_reference : str
            Data reference associated with this argument.
        """

        if name in self.arguments:
            raise exceptions.InvalidArgumentValueError("Argument with name '{name}' already exists.".format(name=name))

        if argument_type not in [ArgumentType.CONTAINER, ArgumentType.DATA]:
            raise exceptions.InvalidArgumentValueError("Invalid argument type: {argument_type}".format(argument_type=argument_type))

        if self.primitive is not None:
            argument_metadata = self.primitive.metadata.query()['primitive_code'].get('arguments', {}).get(name, None)

            if argument_metadata is None:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown argument name '{name}' for primitive {primitive}.".format(
                        name=name,
                        primitive=self.primitive,
                    ),
                )

            if argument_metadata['kind'] != metadata_base.PrimitiveArgumentKind.PIPELINE:
                raise exceptions.InvalidArgumentValueError(
                    "Pipelines can provide only pipeline arguments, '{name}' is of kind {kind}.".format(
                        name=name,
                        kind=argument_metadata['kind'],
                    ),
                )

        self.arguments[name] = {
            'type': argument_type,
            'data': data_reference,
        }

    def add_output(self, output_id: str) -> str:
        """
        Define an output from this step.

        Underlying primitive can have multiple produce methods but not all have to be
        defined as outputs of the step.

        Parameters
        ----------
        output_id : str
            A name of the method producing this output.

        Returns
        -------
        str
            Data reference for the output added.
        """

        if output_id in self.outputs:
            raise exceptions.InvalidArgumentValueError("Output with ID '{output_id}' already exists.".format(output_id=output_id))

        if self.primitive is not None:
            method_metadata = self.primitive.metadata.query()['primitive_code'].get('instance_methods', {}).get(output_id, None)

            if method_metadata is None:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown output ID '{output_id}' for primitive {primitive}.".format(
                        output_id=output_id,
                        primitive=self.primitive,
                    ),
                )

            if method_metadata['kind'] != metadata_base.PrimitiveMethodKind.PRODUCE:
                raise exceptions.InvalidArgumentValueError(
                    "Primitives can output only from produce methods, '{output_id}' is of kind {kind}.".format(
                        output_id=output_id,
                        kind=method_metadata['kind'],
                    ),
                )

        self.outputs.append(output_id)

        return 'steps.{i}.{output_id}'.format(i=self.index, output_id=output_id)

    def add_hyperparameter(self, name: str, argument_type: typing.Any, data: typing.Any) -> None:
        """
        Associate a value for a hyper-parameter of this step (and underlying primitive).

        Parameters
        ----------
        name : str
            Hyper-parameter name.
        argument_type : ArgumentType
            Argument type.
        data : Any
            Data reference associated with this hyper-parameter, or list of data references, or value itself.
        """
        if name in self.hyperparams:
            raise exceptions.InvalidArgumentValueError("Hyper-parameter with name '{name}' already exists.".format(name=name))

        if self.primitive is not None:
            hyperparams = self.get_primitive_hyparparams()

            if name not in hyperparams.configuration:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown hyper-parameter name '{name}' for primitive {primitive}.".format(
                        name=name,
                        primitive=self.primitive,
                    ),
                )

            if argument_type == ArgumentType.VALUE:
                hyperparams.configuration[name].validate(data)

        if argument_type in [ArgumentType.DATA, ArgumentType.PRIMITIVE]:
            if isinstance(data, typing.Sequence):
                if not len(data):
                    raise exceptions.InvalidArgumentValueError("An empty list of hyper-paramater values.")
                if utils.has_duplicates(data):
                    raise exceptions.InvalidArgumentValueError("A list of hyper-paramater values contains duplicate items.")

        self.hyperparams[name] = {
            'type': argument_type,
            'data': data,
        }

    def add_user(self, user_description: typing.Dict) -> None:
        """
        Add a description of user to a list of users associated with the primitive.

        Parameters
        ----------
        user_description : Dict
            User description.
        """

        if 'id' not in user_description:
            raise exceptions.InvalidArgumentValueError("User description is missing user ID.")

        self.users.append(user_description)

    def check_add(self, existing_steps: typing.Sequence[StepBase], available_data_references: typing.AbstractSet[str]) -> None:
        # Order of steps can be arbitrary during execution (given that inputs for a step are available), but we still
        # want some partial order during construction. We want that arguments can already be satisfied by existing steps.
        for argument_description in self.arguments.values():
            if argument_description['data'] not in available_data_references:
                raise exceptions.InvalidPipelineError("Argument data reference '{data_reference}' is not among available data references.".format(data_reference=argument_description['data']))

        for hyperparameter_description in self.hyperparams.values():
            if hyperparameter_description['type'] == ArgumentType.DATA:
                if not isinstance(hyperparameter_description['data'], typing.Sequence):
                    data_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])
                else:
                    data_references = hyperparameter_description['data']
                for data_reference in data_references:
                    if data_reference not in available_data_references:
                        raise exceptions.InvalidPipelineError("Hyper-parameter data reference '{data_reference}' is not among available data references.".format(
                            data_reference=data_reference,
                        ))
            elif hyperparameter_description['type'] == ArgumentType.PRIMITIVE:
                if not isinstance(hyperparameter_description['data'], typing.Sequence):
                    primitive_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])
                else:
                    primitive_references = hyperparameter_description['data']
                for primitive_reference in primitive_references:
                    if not 0 <= primitive_reference < len(existing_steps):
                        raise exceptions.InvalidPipelineError("Invalid primitive reference in a step: {primitive}".format(primitive=primitive_reference))
            elif hyperparameter_description['type'] == ArgumentType.CONTAINER:
                if hyperparameter_description['data'] not in available_data_references:
                    raise exceptions.InvalidPipelineError("Hyper-parameter data reference '{data_reference}' is not among available data references.".format(
                        data_reference=hyperparameter_description['data'],
                    ))
            elif hyperparameter_description['type'] == ArgumentType.VALUE:
                # "VALUE" hyper-parameter value has already been checked in "add_hyperparameter".
                pass
            else:
                assert False, hyperparameter_description['type']

        if self.primitive is not None:
            primitive_arguments = self.primitive.metadata.query()['primitive_code'].get('arguments', {})
            required_arguments_set = {
                argument_name for argument_name, argument in primitive_arguments.items() if 'default' not in argument and argument['kind'] == metadata_base.PrimitiveArgumentKind.PIPELINE
            }

            arguments_set = set(self.arguments.keys())

            missing_arguments_set = required_arguments_set - arguments_set
            if len(missing_arguments_set):
                raise exceptions.InvalidArgumentValueError(
                    "Not all required arguments are provided for the primitive: {missing_arguments_set}".format(
                        missing_arguments_set=missing_arguments_set,
                    )
                )

    def get_primitive_hyparparams(self) -> hyperparams.Hyperparams:
        if self.primitive is None:
            raise exceptions.InvalidStateError("Primitive has not been resolved.")

        return self.primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

    def get_free_hyperparms(self) -> typing.Dict:
        """
        Returns all primitive's hyper-parameters which have not been fixed by the pipeline.

        Returns
        -------
        Dict
            Hyper-parameters configuration for free hyper-parameters.
        """

        free_hyperparams = copy.copy(self.get_primitive_hyparparams().configuration)

        for hyperparam in self.hyperparams:
            del free_hyperparams[hyperparam]

        return free_hyperparams

    def get_output_data_references(self) -> typing.AbstractSet[str]:
        data_references = set()

        for output_id in self.outputs:
            data_references.add('steps.{i}.{output_id}'.format(i=self.index, output_id=output_id))

        return data_references

    @classmethod
    def from_json_structure(cls: typing.Type[SP], step_description: typing.Dict, *, resolver: Resolver = None) -> SP:
        step = cls(step_description['primitive'], resolver=resolver)

        for argument_name, argument_description in step_description['arguments'].items():
            argument_type = ArgumentType[argument_description['type']]
            step.add_argument(argument_name, argument_type, argument_description['data'])

        for output_description in step_description['outputs']:
            step.add_output(output_description['id'])

        for hyperparameter_name, hyperparameter_description in step_description.get('hyperparams', {}).items():
            argument_type = ArgumentType[hyperparameter_description['type']]

            # If "primitive" is not available, we do not parse the value and we leave it in its JSON form.
            if argument_type == ArgumentType.VALUE and step.primitive is not None:
                hyperparams = step.get_primitive_hyparparams()

                if hyperparameter_name not in hyperparams.configuration:
                    raise exceptions.InvalidArgumentValueError(
                        "Unknown hyper-parameter name '{name}' for primitive {primitive}.".format(
                            name=hyperparameter_name,
                            primitive=step.primitive,
                        ),
                    )

                data = hyperparams.configuration[hyperparameter_name].value_from_json(hyperparameter_description['data'])

            else:
                data = hyperparameter_description['data']

            step.add_hyperparameter(hyperparameter_name, argument_type, data)

        for user_description in step_description.get('users', []):
            step.add_user(user_description)

        return step

    def _output_to_json_structure(self, output_id: str) -> typing.Dict:
        return {'id': output_id}

    def _hyperparameter_to_json_structure(self, hyperparameter_name: str) -> typing.Dict:
        hyperparameter_description = copy.copy(self.hyperparams[hyperparameter_name])

        hyperparameter_description['type'] = hyperparameter_description['type'].name

        # If "primitive" is not available, we have the value already in its JSON form.
        if hyperparameter_description['type'] == ArgumentType.VALUE and self.primitive is not None:
            hyperparams = self.get_primitive_hyparparams()

            if hyperparameter_name not in hyperparams.configuration:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown hyper-parameter name '{name}' for primitive {primitive}.".format(
                        name=hyperparameter_name,
                        primitive=self.primitive,
                    ),
                )

            hyperparameter_description['data'] = hyperparams.configuration[hyperparameter_name].value_to_json(hyperparameter_description['data'])

        return hyperparameter_description

    def _argument_to_json_structure(self, argument_name: str) -> typing.Dict:
        argument_description = copy.copy(self.arguments[argument_name])

        argument_description['type'] = argument_description['type'].name

        return argument_description

    def to_json_structure(self) -> typing.Dict:
        step_description = {
            'type': PipelineStep.PRIMITIVE.name,
            'arguments': {argument_name: self._argument_to_json_structure(argument_name) for argument_name in self.arguments.keys()},
            'outputs': [self._output_to_json_structure(output_id) for output_id in self.outputs]
        }

        if self.primitive is None:
            step_description['primitive'] = self.primitive_description
        else:
            primitive_metadata = self.primitive.metadata.query()
            step_description['primitive'] = {
                'id': primitive_metadata['id'],
                'version': primitive_metadata['version'],
                'python_path': primitive_metadata['python_path'],
                'name': primitive_metadata['name'],
            }

            if 'digest' in primitive_metadata:
                step_description['primitive']['digest'] = primitive_metadata['digest']

        if self.hyperparams:
            hyperparams = {}

            for hyperparameter_name in self.hyperparams.keys():
                hyperparams[hyperparameter_name] = self._hyperparameter_to_json_structure(hyperparameter_name)

            step_description['hyperparams'] = hyperparams

        if self.users:
            step_description['users'] = self.users

        return step_description


SS = typing.TypeVar('SS', bound='SubpipelineStep')


class SubpipelineStep(StepBase):
    def __init__(self, pipeline_id: str, *, resolver: Resolver = None) -> None:
        super().__init__(resolver=resolver)

        self.pipeline_id = pipeline_id
        self.pipeline = self.resolver.get_pipeline(pipeline_id)

        self.inputs: typing.List[str] = []
        self.outputs: typing.List[typing.Union[str, None]] = []

    def add_input(self, data_reference: str) -> None:
        if self.pipeline is not None:
            if len(self.inputs) == len(self.pipeline.inputs):
                raise exceptions.InvalidArgumentValueError("All pipeline's inputs are already provided.")

        self.inputs.append(data_reference)

    def add_output(self, output_id: typing.Union[str, None]) -> typing.Union[str, None]:
        """
        Define an output from this step.

        Underlying pipeline can have multiple outputs but not all have to be
        defined as outputs of the step. They can be skipped using ``None``.

        Parameters
        ----------
        output_id : Union[str, None]
            ID to be used in the data reference, mapping pipeline's outputs in order.
            If ``None`` this pipeline's output is ignored and not mapped to a data reference.

        Returns
        -------
        Union[str, None]
            Data reference for the output added. Or ``None`` if output marked to be ignored.
        """

        if output_id is not None:
            if output_id in self.outputs:
                raise exceptions.InvalidArgumentValueError("Output with ID '{output_id}' already exists.".format(output_id=output_id))

        if self.pipeline is not None:
            if len(self.outputs) == len(self.pipeline.outputs):
                raise exceptions.InvalidArgumentValueError("All pipeline's outputs are already mapped.")

        self.outputs.append(output_id)

        if output_id is not None:
            return 'steps.{i}.{output_id}'.format(i=self.index, output_id=output_id)
        else:
            return None

    def check_add(self, existing_steps: 'typing.Sequence[StepBase]', available_data_references: typing.AbstractSet[str]) -> None:
        # Order of steps can be arbitrary during execution (given that inputs for a step are available), but we still
        # want some partial order during construction. We want that arguments can already be satisfied by existing steps.
        for data_reference in self.inputs:
            if data_reference not in available_data_references:
                raise exceptions.InvalidPipelineError("Input data reference '{data_reference}' is not among available data references.".format(data_reference=data_reference))

        # TODO: Check that all inputs are satisfied?

    def get_output_data_references(self) -> typing.AbstractSet[str]:
        data_references = set()

        for output_id in self.outputs:
            if output_id is not None:
                data_references.add('steps.{i}.{output_id}'.format(i=self.index, output_id=output_id))

        return data_references

    @classmethod
    def from_json_structure(cls: typing.Type[SS], step_description: typing.Dict, *, resolver: Resolver = None) -> SS:
        step = cls(step_description['pipeline']['id'], resolver=resolver)

        for input_description in step_description['inputs']:
            step.add_input(input_description['data'])

        for output_description in step_description['outputs']:
            step.add_output(output_description.get('id', None))

        return step

    def _input_to_json_structure(self, data_reference: str) -> typing.Dict:
        return {'data': data_reference}

    def _output_to_json_structure(self, output_id: typing.Union[str, None]) -> typing.Dict:
        if output_id is None:
            return {}
        else:
            return {'id': output_id}

    def to_json_structure(self) -> typing.Dict:
        step_description = {
            'type': PipelineStep.SUBPIPELINE.name,
            'pipeline': {
                'id': self.pipeline_id,
            },
            'inputs': [self._input_to_json_structure(data_reference) for data_reference in self.inputs],
            'outputs': [self._output_to_json_structure(output_id) for output_id in self.outputs],
        }

        return step_description


SL = typing.TypeVar('SL', bound='PlaceholderStep')


class PlaceholderStep(StepBase):
    def __init__(self, resolver: Resolver = None) -> None:
        super().__init__(resolver=resolver)

        self.inputs: typing.List[str] = []
        self.outputs: typing.List[str] = []

    def add_input(self, data_reference: str) -> None:
        self.inputs.append(data_reference)

    def add_output(self, output_id: str) -> str:
        if output_id in self.outputs:
            raise exceptions.InvalidArgumentValueError("Output with ID '{output_id}' already exists.".format(output_id=output_id))

        self.outputs.append(output_id)

        return 'steps.{i}.{output_id}'.format(i=self.index, output_id=output_id)

    def check_add(self, existing_steps: 'typing.Sequence[StepBase]', available_data_references: typing.AbstractSet[str]) -> None:
        # Order of steps can be arbitrary during execution (given that inputs for a step are available), but we still
        # want some partial order during construction. We want that arguments can already be satisfied by existing steps.
        for data_reference in self.inputs:
            if data_reference not in available_data_references:
                raise exceptions.InvalidArgumentValueError("Input data reference '{data_reference}' is not among available data references.".format(data_reference=data_reference))

    def get_output_data_references(self) -> typing.AbstractSet[str]:
        data_references = set()

        for output_id in self.outputs:
            if output_id is not None:
                data_references.add('steps.{i}.{output_id}'.format(i=self.index, output_id=output_id))

        return data_references

    @classmethod
    def from_json_structure(cls: typing.Type[SL], step_description: typing.Dict, *, resolver: Resolver = None) -> SL:
        step = cls(resolver=resolver)

        for input_description in step_description['inputs']:
            step.add_input(input_description['data'])

        for output_description in step_description['outputs']:
            step.add_output(output_description['id'])

        return step

    def _input_to_json_structure(self, data_reference: str) -> typing.Dict:
        return {'data': data_reference}

    def _output_to_json_structure(self, output_id: str) -> typing.Dict:
        return {'id': output_id}

    def to_json_structure(self) -> typing.Dict:
        step_description = {
            'type': PipelineStep.PLACEHOLDER.name,
            'inputs': [self._input_to_json_structure(data_reference) for data_reference in self.inputs],
            'outputs': [self._output_to_json_structure(output_id) for output_id in self.outputs],
        }

        return step_description


P = typing.TypeVar('P', bound='Pipeline')


class Pipeline:
    """
    Class representing a pipeline.

    Attributes
    ----------
    id : str
        An unique ID to identify this pipeline.
    context : PipelineContext
        In which context was the pipeline made.
    created : datetime
        Timestamp of pipeline creation in UTC timezone.
    source : Dict
        Description of source.
    name : str
        Name of the pipeline.
    description : str
        Description of the pipeline.
    users : Sequence[Dict]
        Users associated with the pipeline.
    inputs : Sequence[Dict]
        A sequence of input descriptions which provide names for pipeline inputs.
    outputs : Sequence[Dict]
        A sequence of output descriptions which provide data references for pipeline outputs.
    steps : Sequence[StepBase]
        A sequence of steps defining this pipeline.

    Parameters
    ----------
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

    def __init__(self, pipeline_id: str = None, *, context: typing.Any, created: datetime.datetime = None,
                 source: typing.Dict = None, name: str = None, description: str = None) -> None:
        if pipeline_id is None:
            pipeline_id = str(uuid.uuid4())

        if created is None:
            created = datetime.datetime.now(datetime.timezone.utc)
        elif created.tzinfo is None or created.tzinfo.utcoffset(created) is None:
            raise exceptions.InvalidArgumentValueError("'created' timestamp is missing timezone information.")
        else:
            # Convert to UTC timezone and set "tzinfo" to "datetime.timezone.utc".
            created = created.astimezone(datetime.timezone.utc)

        self.id = pipeline_id
        self.context = context
        self.created = created
        self.source = source
        self.name = name
        self.description = description

        self.inputs: typing.List[typing.Dict] = []
        self.outputs: typing.List[typing.Dict] = []
        self.steps: typing.List[StepBase] = []
        self.users: typing.List[typing.Dict] = []

    def add_input(self, name: str = None) -> str:
        """
        Add an input to the pipeline.

        Parameters
        ----------
        name : str
            Optional human friendly name for the input.

        Returns
        -------
        str
            Data reference for the input added.
        """

        input_description = {}

        if name is not None:
            input_description['name'] = name

        self.inputs.append(input_description)

        return 'inputs.{i}'.format(i=len(self.inputs) - 1)

    def add_output(self, data_reference: str, name: str = None) -> str:
        """
        Add an output to the pipeline.

        Parameters
        ----------
        data_reference : str
            Data reference to use as an output.
        name : str
            Optional human friendly name for the output.

        Returns
        -------
        str
            Data reference for the output added.
        """

        if data_reference not in self.get_available_data_references():
            raise exceptions.InvalidArgumentValueError("Invalid data reference '{data_reference}'.".format(data_reference=data_reference))

        output_description = {
            'data': data_reference,
        }

        if name is not None:
            output_description['name'] = name

        self.outputs.append(output_description)

        return 'outputs.{i}'.format(i=len(self.outputs) - 1)

    def add_step(self, step: StepBase) -> None:
        """
        Add a step to the sequence of steps in the pipeline.

        Parameters
        ----------
        step : StepBase
            A step to add.
        """

        if not isinstance(step, StepBase):
            raise exceptions.InvalidArgumentTypeError("Step is not an instance of StepBase.")

        step.set_index(len(self.steps))

        try:
            step.check_add(self.steps, self.get_available_data_references())
        except Exception as error:
            raise exceptions.InvalidArgumentTypeError("Cannot add step {step_index}.".format(step_index=step.index)) from error

        self.steps.append(step)

    def add_user(self, user_description: typing.Dict) -> None:
        """
        Add a description of user to a list of users associated with the pipeline.

        Parameters
        ----------
        user_description : Dict
            User description.
        """

        if 'id' not in user_description:
            raise exceptions.InvalidArgumentValueError("User description is missing user ID.")

        self.users.append(user_description)

    def get_available_data_references(self) -> typing.AbstractSet[str]:
        """
        Returns a set of data references provided by existing steps (and pipeline inputs).

        Those data references can be used by consequent steps as their inputs.

        Returns
        -------
        Set[str]
            A set of data references.
        """

        data_references = set()

        for i, input_description in enumerate(self.inputs):
            data_references.add('inputs.{i}'.format(i=i))

        for step in self.steps:
            output_data_references = step.get_output_data_references()

            existing_data_references = data_references & output_data_references
            if existing_data_references:
                raise exceptions.InvalidPipelineError("Steps have overlapping output data references: {existing_data_references}".format(existing_data_references=existing_data_references))

            data_references.update(output_data_references)

        return data_references

    def check(self, *, allow_placeholders: bool = False, standard_pipeline: bool = True, input_types: typing.Dict[str, type] = None) -> None:
        """
        Check if the pipeline is a valid pipeline.

        Raises an exception if check fails.

        Parameters
        ----------
        allow_placeholders : bool
            Do we allow placeholders in a pipeline?
        standard_pipeline : bool
            Check it as a standard pipeline (inputs are Dataset objects, output is a DataFrame)?
        input_types : typing.Dict[str, type]
            A map of types available as inputs. If provided, overrides ``standard_pipeline``.
        """

        self._check(allow_placeholders, standard_pipeline, input_types)

    def _check(self, allow_placeholders: bool, standard_pipeline: bool, input_types: typing.Optional[typing.Dict[str, type]]) -> typing.Sequence[TypeInfo]:
        # Generating JSON also checks it against the pipeline schema.
        self.to_json_structure()

        # Map between available data references and their types.
        environment: typing.Dict[str, TypeInfo] = {}

        # Inputs are never singleton.
        if input_types is not None:
            for data_reference, structural_type in input_types.items():
                environment[data_reference] = TypeInfo(structural_type, False)
        elif standard_pipeline:
            for i, input_description in enumerate(self.inputs):
                environment['inputs.{i}'.format(i=i)] = TypeInfo(container.Dataset, False)
        else:
            for i, input_description in enumerate(self.inputs):
                # We do not really know what the inputs are.
                environment['inputs.{i}'.format(i=i)] = TypeInfo(typing.Any, False)  # type: ignore

        for step_index, step in enumerate(self.steps):
            assert step_index == step.index

            if isinstance(step, PlaceholderStep):
                if not allow_placeholders:
                    raise exceptions.InvalidPipelineError("Step {step_index} of pipeline '{pipeline_id}' is a placeholder but there should be no placeholders.".format(
                        step_index=step_index,
                        pipeline_id=self.id,
                    ))

                for data_reference in step.inputs:
                    # This is checked already during pipeline construction in "check_add".
                    assert data_reference in environment

                for data_reference in step.get_output_data_references():
                    # This is checked already during pipeline construction in "add_output".
                    assert data_reference not in environment

                    # We cannot really know a type of the placeholder output given current pipeline description.
                    environment[data_reference] = TypeInfo(typing.Any, None)  # type: ignore

            elif isinstance(step, SubpipelineStep):
                subpipeline_input_types: typing.Dict[str, type] = {}
                for i, data_reference in enumerate(step.inputs):
                    # This is checked already during pipeline construction in "check_add".
                    assert data_reference in environment

                    input_data_reference = 'inputs.{i}'.format(i=i)

                    assert input_data_reference not in subpipeline_input_types
                    subpipeline_input_types[input_data_reference] = environment[data_reference].structural_type

                # Resolving is optional. Of course full checking is not really possible without resolving.
                if step.pipeline is not None:
                    outputs_types = step.pipeline._check(allow_placeholders, False, subpipeline_input_types)

                for i, output_id in enumerate(step.outputs):
                    if output_id is not None:
                        output_data_reference = 'steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)

                        # This is checked already during pipeline construction in "add_output".
                        assert output_data_reference not in environment

                        if step.pipeline is not None:
                            environment[output_data_reference] = outputs_types[i]
                        else:
                            # We cannot really know a type of the output without resolving.
                            environment[output_data_reference] = TypeInfo(typing.Any, None)  # type: ignore

            elif isinstance(step, PrimitiveStep):
                if step.primitive is not None:
                    primitive_metadata = step.primitive.metadata.query()
                    primitive_methods = primitive_metadata['primitive_code'].get('instance_methods', {})
                    primitive_arguments = primitive_metadata['primitive_code'].get('arguments', {})

                for argument_name, argument_description in step.arguments.items():
                    # This is checked already during pipeline construction in "check_add".
                    assert argument_description['data'] in environment

                    if step.primitive is not None:
                        # This is checked already during pipeline construction in "add_argument".
                        assert argument_name in primitive_arguments

                    type_info = environment[argument_description['data']]

                    if argument_description['type'] == ArgumentType.DATA:
                        # The error is only if it is exactly "False". If it is "None", we do not know and we do not want any false positives.
                        if type_info.singleton == False:  # noqa
                            raise exceptions.InvalidPipelineError(
                                "Argument '{argument_name}' of step {step_index} of pipeline '{pipeline_id}' is singleton data, but available data reference is not.".format(
                                    argument_name=argument_name,
                                    step_index=step_index,
                                    pipeline_id=self.id,
                                ),
                            )

                        # We cannot really check if types match because for most container types (except "List")
                        # we do not know the type of elements from just container structural type.
                    elif step.primitive is not None:
                        if type_info.structural_type is typing.Any or primitive_arguments[argument_name]['type'] is typing.Any:
                            # No type information.
                            pass
                        elif not utils.is_subclass(type_info.structural_type, primitive_arguments[argument_name]['type']):
                            raise exceptions.InvalidPipelineError(
                                "Argument '{argument_name}' of step {step_index} of pipeline '{pipeline_id}' has type '{argument_type}', but it is getting a type '{input_type}'.".format(
                                    argument_name=argument_name,
                                    step_index=step_index,
                                    pipeline_id=self.id,
                                    argument_type=primitive_arguments[argument_name]['type'],
                                    input_type=type_info.structural_type,
                                ),
                            )

                if step.primitive is not None:
                    hyperparams = step.get_primitive_hyparparams()

                    for hyperparameter_name, hyperparameter_description in step.hyperparams.items():
                        # This is checked already during pipeline construction in "add_hyperparameter".
                        assert hyperparameter_name in hyperparams.configuration

                        if hyperparameter_description['type'] == ArgumentType.DATA:
                            if not isinstance(hyperparameter_description['data'], typing.Sequence):
                                data_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])
                            else:
                                data_references = hyperparameter_description['data']

                            for data_reference in data_references:
                                # This is checked already during pipeline construction in "check_add".
                                assert data_reference in environment

                                type_info = environment[data_reference]

                                # The error is only if it is exactly "False". If it is "None", we do not know and we do not want any false positives.
                                if type_info.singleton == False:  # noqa
                                    raise exceptions.InvalidPipelineError(
                                        "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' is singleton data, but available data reference '{data_reference}' is not.".format(  # noqa
                                            hyperparameter_name=hyperparameter_name,
                                            step_index=step_index,
                                            pipeline_id=self.id,
                                            data_reference=data_reference,
                                        ),
                                    )

                                # We cannot really check if types match because for most container types (except "List")
                                # we do not know the type of elements from just container structural type.

                        elif hyperparameter_description['type'] == ArgumentType.PRIMITIVE:
                            if not isinstance(hyperparameter_description['data'], typing.Sequence):
                                primitive_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])
                            else:
                                primitive_references = hyperparameter_description['data']

                            primitives = []
                            for primitive_reference in primitive_references:
                                # This is checked already during pipeline construction in "check_add".
                                assert 0 <= primitive_reference < step_index

                                primitive_step = self.steps[primitive_reference]

                                if not isinstance(primitive_step, PrimitiveStep):
                                    raise exceptions.InvalidPipelineError(
                                        "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' does not point to a primitve step (step {primitive_reference}).".format(  # noqa
                                            hyperparameter_name=hyperparameter_name,
                                            step_index=step_index,
                                            pipeline_id=self.id,
                                            primitive_reference=primitive_reference,
                                        ),
                                    )

                                if primitive_step.primitive is None:
                                    primitives.append(typing.Any)
                                else:
                                    primitives.append(primitive_step.primitive)

                            if not isinstance(hyperparameter_description['data'], typing.Sequence):
                                assert len(primitives) == 1

                                if not hyperparams.configuration[hyperparameter_name].can_accept_value_type(primitives[0]):
                                    raise exceptions.InvalidPipelineError(
                                        "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' cannot accept a primitive '{primitive}'.".format(
                                            hyperparameter_name=hyperparameter_name,
                                            step_index=step_index,
                                            pipeline_id=self.id,
                                            primitive=primitives[0],
                                        ),
                                    )
                            else:
                                if not hyperparams.configuration[hyperparameter_name].can_accept_value_type(primitives):
                                    raise exceptions.InvalidPipelineError(
                                        "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' cannot accept primitives {primitives}.".format(
                                            hyperparameter_name=hyperparameter_name,
                                            step_index=step_index,
                                            pipeline_id=self.id,
                                            primitives=primitives,
                                        ),
                                    )

                        elif hyperparameter_description['type'] == ArgumentType.CONTAINER:
                            # This is checked already during pipeline construction in "check_add".
                            assert hyperparameter_description['data'] in environment

                            type_info = environment[hyperparameter_description['data']]

                            if not hyperparams.configuration[hyperparameter_name].can_accept_value_type(type_info.structural_type):
                                raise exceptions.InvalidPipelineError(
                                    "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' cannot accept a value of type '{input_type}'.".format(
                                        hyperparameter_name=hyperparameter_name,
                                        step_index=step_index,
                                        pipeline_id=self.id,
                                        input_type=type_info.structural_type,
                                    ),
                                )

                        elif hyperparameter_description['type'] == ArgumentType.VALUE:
                            # "VALUE" hyper-parameter value has already been checked in "add_hyperparameter".
                            pass

                        else:
                            assert False, hyperparameter_description['type']

                for output_id in step.outputs:
                    output_data_reference = 'steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)

                    assert output_data_reference not in environment

                    if step.primitive is not None:
                        # This is checked already during pipeline construction in "add_output".
                        assert output_id in primitive_methods

                        method_description = primitive_methods[output_id]

                        produce_type = method_description['returns']

                        # This should be checked by some other part of the code (like primitive validation).
                        assert issubclass(produce_type, base.CallResult), produce_type

                        output_type = utils.get_type_arguments(produce_type)[base.T]  # type: ignore

                        environment[output_data_reference] = TypeInfo(output_type, method_description.get('singleton', False))
                    else:
                        # We cannot really know a type of the output without resolving.
                        environment[output_data_reference] = TypeInfo(typing.Any, None)  # type: ignore

        outputs_types = []
        for output_description in self.outputs:
            # This is checked already during pipeline construction in "add_output".
            assert output_description['data'] in environment, output_description['data']

            outputs_types.append(environment[output_description['data']])

        return outputs_types

    @classmethod
    def from_yaml(cls: typing.Type[P], string_or_file: typing.Union[str, typing.TextIO], *, resolver: Resolver = None) -> P:
        description = yaml.load(string_or_file)

        return cls.from_json_structure(description, resolver=resolver)

    @classmethod
    def from_json(cls: typing.Type[P], string_or_file: typing.Union[str, typing.TextIO], *, resolver: Resolver = None) -> P:
        if isinstance(string_or_file, str):
            description = json.loads(string_or_file)
        else:
            description = json.load(string_or_file)

        return cls.from_json_structure(description, resolver=resolver)

    @classmethod
    def _get_step_class(cls, step_type: typing.Any) -> StepBase:
        if step_type == PipelineStep.PRIMITIVE:
            return PrimitiveStep
        elif step_type == PipelineStep.SUBPIPELINE:
            return SubpipelineStep
        elif step_type == PipelineStep.PLACEHOLDER:
            return PlaceholderStep
        else:
            raise exceptions.InvalidArgumentValueError("Invalid step type '{step_type}'.".format(step_type=step_type))

    @classmethod
    def _get_context(cls, pipeline_description: typing.Dict) -> typing.Any:
        return PipelineContext[pipeline_description['context']]

    @classmethod
    def _get_source(cls, pipeline_description: typing.Dict) -> typing.Optional[typing.Dict]:
        return pipeline_description.get('source', None)

    @classmethod
    def from_json_structure(cls: typing.Type[P], pipeline_description: typing.Dict, *, resolver: Resolver = None) -> P:
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
            step = cls._get_step_class(step_description['type']).from_json_structure(step_description, resolver=resolver)
            pipeline.add_step(step)

        for output_description in pipeline_description['outputs']:
            pipeline.add_output(output_description['data'], output_description.get('name', None))

        for user_description in pipeline_description.get('users', []):
            pipeline.add_user(user_description)

        return pipeline

    def _context_to_json_structure(self) -> typing.Any:
        return self.context.name

    def _inputs_to_json_structure(self) -> typing.Sequence[typing.Dict]:
        return self.inputs

    def _outputs_to_json_structure(self) -> typing.Sequence[typing.Dict]:
        return self.outputs

    def _source_to_json_structure(self) -> typing.Optional[typing.Dict]:
        return self.source

    def _users_to_json_structure(self) -> typing.Optional[typing.Sequence[typing.Dict]]:
        # Returns "None" if an empty list.
        return self.users or None

    def to_json_structure(self) -> typing.Dict:
        # Timestamp should already be in UTC and in particular "tzinfo" should be "datetime.timezone.utc".
        assert self.created.tzinfo == datetime.timezone.utc
        # We remove timezone information before formatting to not have "+00:00" added and
        # we then manually add "Z" instead (which has equivalent meaning).
        created = self.created.replace(tzinfo=None).isoformat('T') + 'Z'

        pipeline_description: typing.Dict = {
            'id': self.id,
            'schema': PIPELINE_SCHEMA_VERSION,
            'created': created,
            'context': self._context_to_json_structure(),
            'inputs': self._inputs_to_json_structure(),
            'outputs': self._outputs_to_json_structure(),
            'steps': [],
        }

        source = self._source_to_json_structure()
        if source is not None:
            pipeline_description['source'] = source

        users = self._users_to_json_structure()
        if users is not None:
            pipeline_description['users'] = users

        if self.name is not None:
            pipeline_description['name'] = self.name
        if self.description is not None:
            pipeline_description['description'] = self.description

        for step in self.steps:
            pipeline_description['steps'].append(step.to_json_structure())

        PIPELINE_SCHEMA_VALIDATOR.validate(pipeline_description)

        return pipeline_description

    def to_json(self, file: typing.TextIO = None, **kwargs: typing.Any) -> typing.Optional[str]:
        obj = self.to_json_structure()

        if file is None:
            return json.dumps(obj, **kwargs)
        else:
            json.dump(obj, file, **kwargs)
            return None

    def to_yaml(self, file: typing.TextIO = None, **kwargs: typing.Any) -> typing.Optional[str]:
        obj = self.to_json_structure()

        if 'default_flow_style' not in kwargs:
            kwargs['default_flow_style'] = False

        return yaml.dump(obj, stream=file, **kwargs)


def main() -> None:
    logging.basicConfig()

    parser = argparse.ArgumentParser(description="Parse and validate pipelines.")

    parser.add_argument(
        'pipelines', action='store', metavar='PIPELINE', nargs='+',
        help="path to a pipeline file (.json or .yml)",
    )
    parser.add_argument(
        '-n', '--no-resolving', default=False, action='store_true',
        help="do not resolve primitives and pipelines, this prevents validation to be fully done though",
    )
    parser.add_argument(
        '-r', '--strict-resolving', default=False, action='store_true',
        help="fail resolving if a resolved primitive does not fully match specified primitive reference",
    )
    parser.add_argument(
        '-p', '--pipelines_path', action='append', metavar='PATH', dest='pipeline_search_paths',
        help="path to a directory with pipelines to resolve from (<pipeline id>.json and <pipeline id>.yml), can be specified multiple times, has priority over PIPELINES_PATH environment variable",
    )
    parser.add_argument(
        '-c', '--check', default=False, action='store_true',
        help="also validate a pipeline and not just parse it",
    )
    parser.add_argument(
        '-a', '--allow-placeholders', default=False, action='store_true',
        help="allow placeholders in a pipeline",
    )
    parser.add_argument(
        '-t', '--not-standard-pipeline', default=True, action='store_false', dest='standard_pipeline',
        help="allow a pipeline to not have standard inputs and outputs",
    )

    arguments = parser.parse_args()

    if arguments.no_resolving:
        resolver: Resolver = NoResolver()
    else:
        resolver = Resolver(strict_resolving=arguments.strict_resolving, pipeline_search_paths=arguments.pipeline_search_paths)

    for pipeline_path in arguments.pipelines:
        try:
            with open(pipeline_path, 'r') as pipeline_file:
                if pipeline_path.endswith('.yml'):
                    pipeline = Pipeline.from_yaml(pipeline_file, resolver=resolver)
                elif pipeline_path.endswith('.json'):
                    pipeline = Pipeline.from_json(pipeline_file, resolver=resolver)
                else:
                    raise ValueError("Unknown file extension.")
        except Exception as error:
            raise Exception("Unable to parse pipeline: {pipeline_path}".format(pipeline_path=pipeline_path)) from error

        if arguments.check:
            try:
                pipeline.check(allow_placeholders=arguments.allow_placeholders, standard_pipeline=arguments.standard_pipeline)
            except Exception as error:
                raise Exception("Unable to validate pipeline: {pipeline_path}".format(pipeline_path=pipeline_path)) from error

        try:
            pprint.pprint(pipeline.to_json_structure())
        except Exception as error:
            raise Exception("Unable to print pipeline: {pipeline_path}".format(pipeline_path=pipeline_path)) from error


if __name__ == '__main__':
    main()
