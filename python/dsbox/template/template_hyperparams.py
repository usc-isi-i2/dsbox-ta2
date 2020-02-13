import typing

class Hyperparam(typing.Generic[T]):
    def __init__(self, default: T):
        self._default = T

    def default(self) -> T:
        return self._default

    def sample(self, random_state: RandomState = None) -> T:
        return self._default


class Const(Hyperparam[T]):
    def __init__(self, default: T):
        super.__init__(default)


class Choice(Hyperparam[T]):
    def __init__(self, items: typing.List[T]):
        assert len(items) > 0, "Must have at least one choice: %r" % items
        super.__init__(items[0])
        self._choices = items

    def sample(self, random_state: RandomState = None) -> T:
        return random_state.choice(self._choices)

class Range(Hyperparam[T]):
    def  __init__(self, lower: T, upper: typing.optional[T], default: typing.optional[T], *, inclusive: bool = True):
        self._type = type(lower)
        assert self._type in [int, float], "Lower value must be int for float: %r" % lower
        if self._type is float:
            assert upper is not None, "Must specify upper bound for floats"
        if upper is not None:
            assert type(lower) == type(upper), "lower and upper values must be of the same type: %r %r" % (lower, upper)
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
        super.__init__(default)
        self._lower = lower
        self._upper = upper
        self._inclusive = inclusive

    def sample(self, random_state: RandomState = None) -> int:
        if self._type is int:
            if inclusive:
                if upper:
                    return RandomState.random_integer(lower, upper)
                else:
                    return RandomState.random_integer(lower)
            else:
                if upper:
                    return RandomState.randint(lower, upper)
                else:
                    return RandomState.randint(lower)
        else:
            return (self._upper - self._lower) * random_state.rand() + self.lower
