import abc
import bisect
import operator
import random
import typing

import d3m.exceptions as exceptions

from d3m import utils
from d3m.container.dataset import Dataset
from d3m.metadata.base import PrimitiveMetadata
from d3m import pipeline
from d3m.primitive_interfaces.base import PrimitiveBaseMeta
from d3m.runtime import Runtime

from dsbox.schema.problem import optimization_type, OptimizationType

from .template import TemplatePipeline
from .library import TemplateDescription

HYPERPARAMETER_DIRECTIVE: str = 'hyperparameter_directive'

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


class SimpleConfigurationSpace(ConfigurationSpace[T]):
    def __init__(self, dimension_values: typing.Dict[DimensionName, typing.List[T]], *, 
                 dimension_ordering: typing.List[DimensionName] = None, value_weights: typing.Dict[DimensionName, typing.List[float]] = None) -> None:

        if dimension_ordering is not None and set(dimension_values.keys()) == set(dimension_ordering):
            raise exceptions.InvalidArgumentValueError('The keys of dimension_values and dimesion_ordering must be the same')
        
        if value_weights is not None and not set(dimension_values.keys()) == set(value_weights.keys()):
            raise exceptions.InvalidArgumentValueError('The set of keys of dimension_values and value_weights must be the same')

            for key in dimension_values.keys():
                if not len(dimension_values[key]) == len(value_weights[key]):
                    raise exceptions.InvalidArgumentValueError('The length of dimension_values[{}] and values_weights[{}] must be the same'.format(key, key))

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
        
class DimensionalSearch(typing.Generic[T]):
    """
    Search configuration space on dimension at a time.

    Attributes
    ----------
    evaluate : Callable[[typing.Dict], float]
        Evaluate given point in configuration space
    configuration_space: ConfigurationSpace[T]
        Definition of the configuration space
    minimize: bool
        If True, minimize the value returned by `evaluate` function
    """
    def __init__(self, evaluate: typing.Callable[[typing.Dict], float], configuration_space: ConfigurationSpace[T], minimize: bool) -> None:
        self.evaluate = evaluate
        self.configuration_space = configuration_space
        self.minimize = minimize
        self.dimension_ordering = configuration_space.get_dimension_search_ordering()

    def random_assignment(self) -> typing.Dict[str, T]:
        assignment: typing.Dict[str, T] = {}
        for dimension in self.dimension_ordering:
            assignment[dimension] = random.choice(self.configuration_space.get_values(dimension))
        return assignment

    def search_one_iter(self, candidate: typing.Dict[str, T] = None, candidate_value: float = None, max_per_dimension=10):
        """
        Performs one iteration of dimensional search
        """
        if candidate is None:
            candidate = self.random_assignment()

        for dimension in self.dimension_ordering:
            choices: typing.List[T] = self.configuration_space.get_values(dimension)
            weights = [self.configuration_space.get_weight(dimension, x) for x in choices]
            selected = random_choices_without_replacement(choices, weights, max_per_dimension)
            if candidate_value is None and candidate[dimension] in selected:
                selected.remove(candidate[dimension])

            new_candidates = []
            for value in selected:
                new = dict(candidate)
                new[dimension] = value
                new_candidates.append(new)
            values = [self.evaluate(x) for x in new_candidates]
            best_index = values.index(min(values))
            if candidate_value is None:
                candidate = new_candidates[best_index]
                candidate_value = values[best_index]
            elif (self.minimize and values[best_index] < candidate_value) or (not self.minimize and values[best_index] > candidate_value):
                candidate = new_candidates[best_index]
                candidate_value = values[best_index]
        return (candidate, candidate_value)

    def search(self, candidate: typing.Dict[str, T] = None, candidate_value: float = None, num_iter=3, max_per_dimension=10):
        for i in range(num_iter):
            candidate, candidate_value = self.search_one_iter(candidate, candidate_value, max_per_dimension=max_per_dimension)
        return (candidate, candidate_value)

# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

class TemplateDimensionalSearch(DimensionalSearch[PythonPath]):
    """
    Use dimensional search to find best pipeline.

    Attributes
    ----------
    template_description : TemplateDescription
        The template pipeline to be fill in
    configuration_space : ConfigurationSpace[PythonPath]
        Configuration space where values are primitive python paths
    primitive_index : typing.Dict[PythonPath, PrimitiveBaseMeta]
        Map from primitive python paths to primitive class from d3m.index.search()
    train_dataset : Dataset
        The dataset to train pipeline
    validation_dataset : Dataset
        The dataset to evaluate pipeline
    performance_metrics : typing.List[typing.Dict]
        Performance metrics from parse_problem_description()['problem']['performance_metrics']
    resolver : typing.Optional[Resolver]
        Resolve primitives
    """

    def __init__(self, template_description: TemplateDescription,
                 configuration_space: ConfigurationSpace[PythonPath],
                 primitive_index: typing.Dict[PythonPath, PrimitiveBaseMeta],
                 train_dataset : Dataset, validation_dataset : Dataset, 
                 performance_metrics: typing.List[typing.Dict],
                 resolver: Resolver = None) -> None:
        
        # Use first metric from validation
        minimize = optimization_type(performance_metrics[0]['metric']) == OptimizationType.MINIMIZE
        super().__init__(self.evaluate, configuration_space, minimize)

        self.template_description = template_description
        self.template: TemplatePipeline = self.template_description.template
        # self.configuration_space = configuration_space
        self.primitive_index = primitive_index
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.performance_metrics = performance_metrics
        self.resolver = resolver

        if not set(self.template.template_nodes.keys()) <= set(configuration_space.get_dimensions()):
            raise exceptions.InvalidArgumentValueError("Not all template steps are in configuration space: {}".format(self.template.template_nodes.keys()))

    def evaluate(self, configuration: typing.Dict[str, PythonPath]) -> float:
        
        # convert PythonPath to primitive metadata
        metadata_configuration: typing.Dict[str, PrimitiveMetadata] = {
            key: self.primitive_index[python_path].metadata for key, python_path in configuration.items()}

        return self._evaluate(metadata_configuration)

    def _evaluate(self, metadata_configuration: typing.Dict):

        binding = self.template.to_steps(metadata_configuration, self.resolver)
        pipeline = self.template.get_pipeline(binding, None, context='PRETRAINING')

        # Todo: update ResourceManager to run pipeline:  ResourceManager.add_pipeline(pipeline)
        run = Runtime(pipeline)

        run.fit(inputs=[self.train_dataset])
        training_ground_truth = run.fit_outputs[self.template_description.target_step]
        training_prediction = run.fit_outputs[self.template_description.predicted_target_step]

        results = run.produce(inputs=[self.validation_dataset])
        validation_ground_truth = run.produce_outputs[self.template_description.target_step]
        # results == validation_prediction
        validation_prediction = run.produce_outputs[self.template_description.predicted_target_step]

        training_metrics = []
        validation_metrics = []
        for metric_description in self.performance_metrics:
            metric: typing.Callable = metric_description['metric'].get_function()
            params: typing.Dict = metric_description['params']
            training_metrics.append({
                'metric': metric_description['metric'],
                'value': metric(training_ground_truth, training_prediction)
            })        
            validation_metrics.append({
                'metric': metric_description['metric'],
                'value': metric(validation_ground_truth, validation_prediction)
            })

        # Use first metric from validation
        return validation_metrics[0]['value']

class HyperparamDirective(utils.Enum):
    """
    Specify how to choose hyperparameters
    """
    DEFAULT = 1
    RANDOM = 2
    
class HyperparamConfigurationSpace(ConfigurationSpace[PythonPath]) -> ConfigurationSpace[typing.Tuple[PythonPath, Hyperparams]]:
    def __init__(self, configuration_space: ConfigurationSpace[PythonPath],
                 primitive_index: typing.Dict[PythonPath, PrimitiveBaseMeta]):
        self.base_configuration_space = configuration_space
        self.primitive_index = primitive_index
        
    def get_dimension(self):
        return self.base_configuration_space.get_dimensions()
    def get_values(self, dimension: DimensionName) -> typing.List[T]:
        base_values = self._dimension_values[dimension]
        hyperparam_directive = [HyperparamDirective.DEFAULT] + [HyperparamDirective.RANDOM] * 3
        values = [(path, directive) for path in base_values for directive in hyperparam_directive]
        return values
    def get_weight(self, dimension: DimensionName, value: T) -> float:
        return self.base_configuration_space.get_weight(dimension, value)
    def get_dimension_search_ordering(self) -> typing.List[DimensionName]:
        return self.base_configuration_space.get_dimension_search_ordering()

class HyperparameterResolver(pipeline.Resolver):
    def __init__(self, strict_resolving: bool = False) -> None:
        super.__init__(strict_resolving)
        
        def get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[base.PrimitiveBase]:
            primitive = super.get_primitive(primitive_description)
            if primitive is not None and HYPERPARAMETER_DIRECTIVE in primitive_description:
                if primitive_description[HYPERPARAMETER_DIRECTIVE] == HyperparamDirective.RANDOM:
                    hyperparams_class = primitive.getHyperparamClass()
                    primitive.setHyperparams(hyperparams_class.sample())

            return primitive
                                                                
class TemplateDimensionalRandomHyperparameterSearch(TemplateDimensionalSearch):
    """
    Use dimensional search with random hyperparameters to find best pipeline.
    """
    def __init__(self, template_description: TemplateDescription,
                 configuration_space: HyperparamConfigurationSpace,
                 primitive_index: typing.Dict[PythonPath, PrimitiveBaseMeta],
                 train_dataset : Dataset, validation_dataset : Dataset, 
                 performance_metrics: typing.List[typing.Dict],
                 resolver: HyperparameterResolver = None) -> None:
        if resolver is None:
            resolver = HyperparameterResolver()
        super().__init__(template_description, configuration_space, primitive_index,
                         train_dataset, validation_dataset, performance_metrics, resolver)

    def evaluate(self, configuration: typing.Dict[str, typing.Tuple[PythonPath, HyperparamDirective]]) -> float:
        # convert PythonPath to primitive metadata
        metadata_configuration: typing.Dict[str, PrimitiveMetadata] = {}
        for key, (python_path, directive) in configuration.items():
            metadata_configuration[key] = dict(self.primitive_index[python_path].metadata)
            metadata_configuration[key][HYPERPARAMETER_DIRECTIVE] = directive

        return self._evaluate(metadata_configuration)
    
def random_choices_without_replacement(population, weights, k=1):
    """
    Randomly sample multiple element based on weights witout replacement.
    """
    assert len(weights) == len(population)
    if k > len(population):
        k = len(population)
    weights = list(weights)
    result = []
    for index in range(k):
        cum_weights = list(accumulate(weights))
        total = cum_weights[-1]
        i = bisect.bisect(cum_weights, random.random() * total)
        result.append(population[i])
        weights[i] = 0
    return result

def accumulate(iterable, func=operator.add):
    """
    Sum all the elments
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total

