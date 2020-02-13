import abc
import pprint
import random
import typing

from operator import itemgetter

import d3m.exceptions as exceptions

from .template_hyperparams import Hyperparam

DimensionName = typing.NewType('DimensionName', str)


class ConfigurationSpace():
    """
    Defines search space, i.e. possible values for each search dimension, and weight for each value
    """

    @abc.abstractmethod
    def get_dimensions(self) -> typing.List[DimensionName]:
        """
        Returns the dimension names of the configuration space
        """

    # @abc.abstractmethod
    # def get_values(self, dimension: DimensionName) -> typing.List[T]:
    #     """
    #     Returns the values associated with a dimension
    #     """

    # @abc.abstractmethod
    # def get_weight(self, dimension: DimensionName, value: typing.Any) -> float:
    #     """
    #     Returns the wieght associated with each dimension value
    #     """

    # @abc.abstractmethod
    # def get_dimension_search_ordering(self) -> typing.List[DimensionName]:
    #     """
    #     Returns the dimension names in order of search preference
    #     """

    @abc.abstractmethod
    def get_random_assignment(self) -> 'ConfigurationPoint':
        """
        Randomly assigns a value for each dimension
        """

    @abc.abstractmethod
    def get_default_assignment(self) -> 'ConfigurationPoint':
        """
        Assigns default value for each dimension
        """

    def get_point(self, values: typing.Dict[DimensionName, typing.Any]) -> 'ConfigurationPoint':
        """
        Returns the point asscoiate with values.
        """
        return ConfigurationPoint(self, values)


class ConfigurationPoint(typing.Dict[DimensionName, typing.Any]):
    def __init__(self, space: ConfigurationSpace, values: typing.Dict[DimensionName, typing.Any]) -> None:
        super().__init__(values)
        self.space = space
        self.data: typing.Dict = {}


class SimpleConfigurationSpace(ConfigurationSpace):
    '''
    Implementation that explicitly enumerates all configuration space grid points
    '''
    def __init__(self, dimension_values: typing.Dict[DimensionName, typing.List], *,
                 dimension_ordering: typing.List[DimensionName] = None,
                 value_weights: typing.Dict[DimensionName, typing.List[float]] = None) -> None:

        if dimension_ordering is not None and set(dimension_values.keys()) == set(dimension_ordering):
            raise exceptions.InvalidArgumentValueError(
                'The keys of dimension_values and dimesion_ordering must be the same')

        if value_weights is not None and not set(dimension_values.keys()) == set(value_weights.keys()):
            raise exceptions.InvalidArgumentValueError(
                'The set of keys of dimension_values and value_weights must be the same')

            for key in dimension_values.keys():
                if not len(dimension_values[key]) == len(value_weights[key]):
                    raise exceptions.InvalidArgumentValueError(
                        'The length of dimension_values[{}] and values_weights[{}] must be the same'.format(
                            key, key))

        if value_weights is None:
            value_weights = {}
            for key in dimension_values.keys():
                value_weights[key] = [1.0] * len(dimension_values[key])

        if dimension_ordering is None:
            dimension_ordering = list(dimension_values.keys())

        self._dimension_values: typing.Dict[DimensionName, typing.List] = dimension_values
        self._value_weights: typing.Dict[DimensionName, typing.List[float]] = value_weights
        self._dimension_ordering = dimension_ordering

    def get_dimensions(self):
        return list(self._dimension_values.keys())

    def get_values(self, dimension: DimensionName) -> typing.List:
        return self._dimension_values[dimension]

    def get_weight(self, dimension: DimensionName, value: typing.Any) -> float:
        return self._value_weights[dimension][self.get_values(dimension).index(value)]

    def get_dimension_search_ordering(self) -> typing.List[DimensionName]:
        return self._dimension_ordering

    def get_point(self, values: typing.Dict[DimensionName, typing.Any]):
        # TODO: SimpleConfigurationSpace should manage and reuse ConfigurationPoints
        return ConfigurationPoint(self, values)

    def get_first_assignment(self) -> ConfigurationPoint:
        '''
        Assign the first value for each dimension
        '''
        # print(self._dimension_ordering)
        assignment: typing.Dict[DimensionName, typing.Any] = {}
        for dimension in self._dimension_ordering:
            assignment[dimension] = self.get_values(dimension)[0]
            # print(dimension, self.get_values(dimension)[0])
        return ConfigurationPoint(self, assignment)

    def get_default_assignment(self) -> ConfigurationPoint:
        return self.get_first_assignment()

    def get_random_assignment(self) -> ConfigurationPoint:
        """
        Randomly assigns a value for each dimension
        """
        assignment: typing.Dict[DimensionName, typing.Any] = {}
        for dimension in self._dimension_ordering:
            assignment[dimension] = random.choice(self.get_values(dimension))

        return ConfigurationPoint(self, assignment)

    def get_dimension_length(self, kw: DimensionName) -> int:
        """
        Return the length of the list a configuration point
        Args:
            kw:
                name of the dimension
        Returns:

        """
        return len(self.get_values(kw))

    def __str__(self):
        """
        Returns: the configuration point as a human-readable string
        """
        return pprint.pformat(self._dimension_values)

class PrimitiveHyperparams():
    def __init__(self, primitive_name: str, hyperparams: typing.Dict[str, Hyperparam]):
        self.primitive_name = primitive_name
        self.hyperparams = hyperparams

    def get_default_assignment(self) -> typing.Dict:
        hyperparams = {}
        for name, function in self.hyperparams.items():
            hyperparams[name] = function.default()
        result = {
            "primitive": self.primitive_name,
            "hyperparameters": hyperparams
        }
        return result

    def get_random_assignment(self) -> typing.Dict:
        hyperparams = {}
        for name, function in self.hyperparams.items():
            hyperparams[name] = function.sample()
        result = {
            "primitive": self.primitive_name,
            "hyperparameters": hyperparams
        }
        return result

class TemplateStepHyperparams():
    def __init__(self, primitive_hyperaparms: typing.List[PrimitiveHyperparams],
                 primitive_weights: typing.Optional[typing.List[float]]):

        assert len(primitive_hyperaparms) > 0, "Must provided at least one PrimitiveHyperparams"
        assert primitive_weights is None or len(primitive_hyperaparms)==len(primitive_weights), "Must have samme length"

        self.primitive_hyperaparms = primitive_hyperaparms
        self.no_weights_specified = primitive_weights is None
        if primitive_weights is None:
            primitive_weights = [1.0] * len(primitive_hyperaparms)
        self.primitive_weights = primitive_weights

    def get_default_assignment(self) -> typing.Dict:
        if self.no_weights_specified:
            return self.primitive_hyperaparms[0].get_default_assignment()
        sorted_by_weight = sorted(zip(self.primitive_weights, self.primitive_hyperaparms),
                                  key=itemgetter(0), reverse=True)
        return sorted_by_weight[0][1].get_default_assignment()

    def get_random_assignment(self) -> typing.Dict:
        return random.choices(self.primitive_hyperaparms, self.primitive_weights)[0].get_random_assignment()

class ImplicitConfigurationSpace(ConfigurationSpace):
    def __init__(self, conf_space: typing.Dict[DimensionName, typing.List[PrimitiveHyperparams]]):
        self.conf_space = conf_space

    def get_default_assignment(self):
        result = {}
        for (domain_name, step_hyperparams) in self.conf_space.items():
            result[domain_name] = step_hyperparams[0].get_default_assignment()
        return result

    def get_random_assignment(self):
        result = {}
        for (domain_name, step_hyperparams) in self.conf_space.items():
            result[domain_name] = random.choices(step_hyperparams)[0].get_default_assignment()
        return result
