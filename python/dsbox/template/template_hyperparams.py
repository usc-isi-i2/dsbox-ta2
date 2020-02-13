import abc
import typing

import numpy as np
from numpy.random import RandomState

T = typing.TypeVar('T')


class Hyperparam(typing.Generic[T]):
    def __init__(self, default: T):
        self._default = default

    def default(self) -> T:
        return self._default

    @abc.abstractmethod
    def sample(self, random_state: RandomState = None) -> T:
        """
        Returns a random sample.
        """

class Const(Hyperparam[T]):
    """
    Always returns the same value."
    """
    def __init__(self, default: T):
        super().__init__(default)

    def sample(self, random_state: RandomState = None) -> T:
        return self._default

    def __repr__(self):
        return "Const(%r)" % (self._default,)


class Choice(Hyperparam[T]):
    """
    Returns a value from given list of items
    """
    def __init__(self, items: typing.List[T], default: typing.Optional[T] = None):
        assert len(items) > 0, "Must have at least one choice: %r" % items
        if default is None:
            super().__init__(items[0])
        else:
            assert default in items, "List (%s) must contain default (%s)" % (items, default)
            super().__init__(default)
        self._choices = items

    def sample(self, random_state: RandomState = None) -> T:
        return random_state.choice(self._choices)

    def __repr__(self):
        return "Choice(%r, default=%r)" % (self._choices, self._default)


class LogRange(Hyperparam[float]):
    """
    Returns float value between `lower` and `upper`. Use logspace spacing for sampling'
    """
    def  __init__(self, lower: float, upper: float, default: typing.Optional[float] = None):
        assert lower < upper, "lower value must be smaller than upper value: %r < %r" % (lower, upper)
        if default is None:
            default = float(np.exp((np.log(lower) + np.log(upper))/ 2))
        else:
            assert lower <= default, "lower <= default: %r <= %r" % (lower, default)
            assert default <= upper, "default <= upper: %r <= %r" % (default, upper)
        self._lower = lower
        self._upper = upper
        self._log_lower = np.log(lower)
        self._log_upper = np.log(upper)
        super().__init__(default)

    def sample(self, random_state: RandomState) -> float:
        return float(np.exp((self._log_upper - self._log_lower) * random_state.rand() + self._log_lower))

    def __repr__(self):
        return "LogRange(%r, %r, default=%r)" % (self._lower, self._upper, self._default)

class Range(Hyperparam[T]):
    """
    Returns float value between `lower` and `upper`. Use unfiorm spacing for sampling'
    """
    def  __init__(self, lower: T, upper: typing.Optional[T], default: typing.Optional[T] = None, inclusive: bool = True):
        self._type = type(lower)
        assert self._type in [int, float], "Lower value must be int for float: %r" % lower
        if self._type is float:
            assert upper is not None, "Must specify upper bound for floats"
        if upper is not None:
            assert type(lower) == type(upper), "lower and upper values must be of the same type: %r %r" % (lower, upper)
            assert lower < upper, "lower value must be smaller than upper value: %r < %r" % (lower, upper)
        if default is None:
            if upper is None:
                default = lower
            else:
                if self._type is int:
                    default = int((lower + upper) / 2)
                else:
                    default = (lower + upper) / 2
        else:
            assert type(lower) == type(default), "lower and default values must be of the same type: %r %r" % (lower, default)
        super().__init__(default)
        self._lower = lower
        self._upper = upper
        self._inclusive = inclusive

    def sample(self, random_state: RandomState = None) -> int:
        if self._type is int:
            if self._inclusive:
                if self._upper:
                    return RandomState.random_integer(self._lower, self._upper)
                else:
                    return RandomState.random_integer(self._lower)
            else:
                if self._upper:
                    return RandomState.randint(self._lower, self._upper)
                else:
                    return RandomState.randint(self._lower)
        else:
            return (self._upper - self._lower) * random_state.rand() + self._lower

    def __repr__(self):
        return "Range(%r, %r, default=%r, inclusive=%r)" % (self._lower, self._upper, self._default, self._inclusive)
