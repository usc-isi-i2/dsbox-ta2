import abc
import typing

import pprint

import d3m.exceptions as exceptions
from random import choice
T = typing.TypeVar('T')
DimensionName = typing.NewType('DimensionName', str)


class ConfigurationSpace(typing.Generic[T]):
    """
    Defines search space, i.e. possible values for each search dimension, and weight for each value
    """

    @abc.abstractmethod
    def get_dimensions(self) -> typing.List[DimensionName]:
        """
        Returns the dimension names of the configuration space
        """
        pass

    @abc.abstractmethod
    def get_values(self, dimension: DimensionName) -> typing.List[T]:
        """
        Returns the values associated with a dimension
        """
        pass

    @abc.abstractmethod
    def get_weight(self, dimension: DimensionName, value: T) -> float:
        """
        Returns the wieght associated with each dimension value
        """
        pass

    @abc.abstractmethod
    def get_dimension_search_ordering(self) -> typing.List[DimensionName]:
        """
        Returns the dimension names in order of search preference
        """
        pass

    def get_point(self, values: typing.Dict[DimensionName, T]):
        """
        Returns the point asscoiate with values.
        """
        return ConfigurationPoint(self, values)



class ConfigurationPoint(typing.Dict[DimensionName, T]):
    def __init__(self, space: ConfigurationSpace[T], values: typing.Dict[DimensionName, T]) -> None:
        super().__init__(values)
        self.space = space
        self.data: typing.Dict = {}


# TODO: SimpleConfigurationSpace should manage and reuse ConfigurationPoints
class SimpleConfigurationSpace(ConfigurationSpace[T]):
    def __init__(self, dimension_values: typing.Dict[DimensionName, typing.List[T]], *,
                 dimension_ordering: typing.List[DimensionName] = None, value_weights: typing.Dict[DimensionName, typing.List[float]] = None) -> None:

        if dimension_ordering is not None and set(dimension_values.keys()) == set(dimension_ordering):
            raise exceptions.InvalidArgumentValueError(
                'The keys of dimension_values and dimesion_ordering must be the same')

        if value_weights is not None and not set(dimension_values.keys()) == set(value_weights.keys()):
            raise exceptions.InvalidArgumentValueError(
                'The set of keys of dimension_values and value_weights must be the same')

            for key in dimension_values.keys():
                if not len(dimension_values[key]) == len(value_weights[key]):
                    raise exceptions.InvalidArgumentValueError(
                        'The length of dimension_values[{}] and values_weights[{}] must be the same'.format(key, key))

        if value_weights is None:
            value_weights = {}
            for key in dimension_values.keys():
                value_weights[key] = [1.0] * len(dimension_values[key])

        if dimension_ordering is None:
            dimension_ordering = list(dimension_values.keys())

        self._dimension_values: typing.Dict[DimensionName, typing.List[T]] = dimension_values
        self._value_weights: typing.Dict[DimensionName, typing.List[float]] = value_weights
        self._dimension_ordering = dimension_ordering

    def get_dimensions(self):
        return list(self._dimension_values.keys())

    def get_values(self, dimension: DimensionName) -> typing.List[T]:
        return self._dimension_values[dimension]

    def get_weight(self, dimension: DimensionName, value: T) -> float:
        return self._value_weights[dimension][self.get_values(dimension).index(value)]

    def get_dimension_search_ordering(self) -> typing.List[DimensionName]:
        return self._dimension_ordering

    def get_point(self, values: typing.Dict[DimensionName, T]):
        # TODO: SimpleConfigurationSpace should manage and reuse ConfigurationPoints
        return ConfigurationPoint(self, values)

    def get_first_assignment(self) -> typing.Dict[DimensionName, T]:
        '''
        Assign the first value for each dimension
        '''
        # print(self._dimension_ordering)
        assignment: typing.Dict[DimensionName, T] = {}
        for dimension in self._dimension_ordering:
            assignment[dimension] = self.get_values(dimension)[0]
            # print(dimension, self.get_values(dimension)[0])
        return ConfigurationPoint(self, assignment)

    def get_random_assignment(self) -> typing.Dict[DimensionName, T]:
        """
        Randomly assigns a value for each dimension
        """
        assignment: typing.Dict[DimensionName, T] = {}
        for dimension in self._dimension_ordering:
            assignment[dimension] = choice(self.get_values(dimension))

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
