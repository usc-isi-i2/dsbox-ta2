import abc
import bisect
import operator
import random
import traceback
import typing

import d3m.exceptions as exceptions

from d3m import utils
from d3m.container.dataset import Dataset
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.hyperparams import Hyperparams
from d3m.metadata.pipeline import Resolver
from d3m.primitive_interfaces.base import PrimitiveBaseMeta, PrimitiveBase
from d3m.runtime import Runtime

from dsbox.schema.problem import optimization_type, OptimizationType

from .template import TemplatePipeline, HYPERPARAMETER_DIRECTIVE, HyperparamDirective
from .library import TemplateDescription

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

    def __init__(self, evaluate: typing.Callable[[ConfigurationPoint[T]], typing.Tuple[float, dict]], 
                 configuration_space: ConfigurationSpace[T], minimize: bool) -> None:
        self.evaluate = evaluate
        self.configuration_space = configuration_space
        self.minimize = minimize
        self.dimension_ordering = configuration_space.get_dimension_search_ordering()

    def random_assignment(self) -> typing.Dict[DimensionName, T]:
        """
        Randomly assigns a value for each dimension
        """
        assignment: typing.Dict[DimensionName, T] = {}
        for dimension in self.dimension_ordering:
            assignment[dimension] = random.choice(
                self.configuration_space.get_values(dimension))
        return assignment

    def search_one_iter(self, candidate: ConfigurationPoint[T] = None, candidate_value: float = None, max_per_dimension=10):
        """
        Performs one iteration of dimensional search.

        Parameters
        ----------
        candidate: ConfigurationPoint[T]
            Current best candidate
        candidate_value: float
            The valude for the current best candidate
        max_per_dimension: int
            Maximunum number of values to search per dimension
        """
        if candidate is None:
            candidate = ConfigurationPoint(
                self.configuration_space, self.random_assignment())

        for dimension in self.dimension_ordering:
            choices: typing.List[T] = self.configuration_space.get_values(dimension)
            weights = [self.configuration_space.get_weight(
                dimension, x) for x in choices]
            selected = random_choices_without_replacement(
                choices, weights, max_per_dimension)

            # No need to evaluate if value is already known
            if candidate_value is None and candidate[dimension] in selected:
                selected.remove(candidate[dimension])

            new_candidates: typing.List[ConfigurationPoint] = []
            for value in selected:
                new = dict(candidate)
                new[dimension] = value
                new_candidates.append(self.configuration_space.get_point(new))
            values = []
            sucessful_candidates = []
            for x in new_candidates:
                try:
                    result = self.evaluate(x)
                    values.append(result[0])
                    print('result={} x.data={}'.format(result[1], x.data))
                    sucessful_candidates.append(x)
                except:
                    print('Pipeline failed: ', x)
                    traceback.print_exc()

            # All primitives failed
            if len(values)==0:
                return (None, None)

            best_index = values.index(min(values))
            if candidate_value is None:
                candidate = sucessful_candidates[best_index]
                candidate_value = values[best_index]
            elif (self.minimize and values[best_index] < candidate_value) or (not self.minimize and values[best_index] > candidate_value):
                candidate = sucessful_candidates[best_index]
                candidate_value = values[best_index]
        # here we can get the details of pipelines from "candidate.data"
        return (candidate, candidate_value)

    def search(self, candidate: ConfigurationPoint[T] = None, candidate_value: float = None, num_iter=3, max_per_dimension=10):
        for i in range(num_iter):
            candidate, candidate_value = self.search_one_iter(candidate, candidate_value, max_per_dimension=max_per_dimension)
            if candidate is None:
                return (None, None)

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
                 train_dataset: Dataset, validation_dataset: Dataset,
                 performance_metrics: typing.List[typing.Dict],
                 resolver: Resolver = None) -> None:

        # Use first metric from validation

        minimize = optimization_type(performance_metrics[0]['metric']) == OptimizationType.MINIMIZE
        super().__init__(self.evaluate_pipeline, configuration_space, minimize)

        self.template_description = template_description
        self.template: TemplatePipeline = self.template_description.template
        # self.configuration_space = configuration_space
        self.primitive_index = primitive_index
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.performance_metrics = performance_metrics
        self.resolver = resolver

        if not set(self.template.template_nodes.keys()) <= set(configuration_space.get_dimensions()):
            raise exceptions.InvalidArgumentValueError(
                "Not all template steps are in configuration space: {}".format(self.template.template_nodes.keys()))

    def evaluate_pipeline(self, configuration: ConfigurationPoint[PythonPath]) -> typing.Tuple[float, dict]:
        """
        Evaluate at configuration point.
        Note: This methods will modify the configuration point, by updating its data field.
        """

        # convert PythonPath to primitive metadata
        metadata_configuration: typing.Dict[DimensionName, PrimitiveMetadata] = {
            key: self.primitive_index[python_path].metadata.query() for key, python_path in configuration.items()}

        value, new_data = self._evaluate(metadata_configuration)
        configuration.data.update(new_data)        
        return value, configuration.data

    def _evaluate(self, metadata_configuration: typing.Dict) -> typing.Tuple[float, dict]:

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

        data = {
            'runtime': run,
            'pipeline': pipeline,
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics
        }
        # Use first metric from validation
        return validation_metrics[0]['value'], data

PythonPathWithHyperaram = typing.Tuple[PythonPath, int, HyperparamDirective]

def generate_hyperparam_configuration_space(space: ConfigurationSpace[PythonPath]) -> ConfigurationSpace[PythonPathWithHyperaram]:
    new_space = {}
    for name in space.get_dimensions():
        values = []
        for path in space.get_values(name):
            for index, hyperparam_directive in enumerate([HyperparamDirective.DEFAULT] + [HyperparamDirective.RANDOM] * 3):
                values.append((path, index, hyperparam_directive))
        new_space[name] = values
    return SimpleConfigurationSpace(new_space)

# Not used            
class HyperparameterResolver(Resolver):
    def __init__(self, strict_resolving: bool = False) -> None:
        super().__init__(strict_resolving)
        
    def get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[PrimitiveBase]:
        primitive = super().get_primitive(primitive_description)
        if primitive is not None and HYPERPARAMETER_DIRECTIVE in primitive_description:
            if primitive_description[HYPERPARAMETER_DIRECTIVE] == HyperparamDirective.RANDOM:
                hyperparams_class = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                primitive.hyperparams(hyperparams_class.sample())

        return primitive
                                                                
class TemplateDimensionalRandomHyperparameterSearch(DimensionalSearch[PythonPathWithHyperaram]):
    """
    Use dimensional search with random hyperparameters to find best pipeline.
    """
    def __init__(self, template_description: TemplateDescription,
                 configuration_space: ConfigurationSpace[PythonPath],
                 primitive_index: typing.Dict[PythonPath, PrimitiveBaseMeta],
                 train_dataset : Dataset, validation_dataset : Dataset, 
                 performance_metrics: typing.List[typing.Dict],
                 resolver: Resolver = None) -> None:
        # Use first metric from validation
        minimize = optimization_type(performance_metrics[0]['metric']) == OptimizationType.MINIMIZE
        # if resolver is None:
        #     resolver = HyperparameterResolver()
        if resolver is None:
            resolver = Resolver()
        hyperparam_configuration_space = generate_hyperparam_configuration_space(configuration_space)
        super().__init__(self.evaluate_pipeline, hyperparam_configuration_space, minimize)
        self.template_description = template_description
        self.template: TemplatePipeline = self.template_description.template
        self.primitive_index = primitive_index
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.performance_metrics = performance_metrics
        self.resolver = resolver

    def evaluate_pipeline(self, point: ConfigurationPoint[PythonPathWithHyperaram]) -> typing.Tuple[float, dict]:
        """
        Evaluate at configuration point.
        Note: This methods will modify the configuration point, by updating its data field.
        """
        # convert PythonPath to primitive metadata
        metadata_configuration: typing.Dict[DimensionName, dict] = {}
        for key, (python_path, index, directive) in point.items():
            metadata_configuration[key] = dict(self.primitive_index[python_path].metadata.query())
            metadata_configuration[key][HYPERPARAMETER_DIRECTIVE] = directive

        value, new_data = self._evaluate(metadata_configuration)
        configuration.data.update(new_data)        
        return value, configuration.data
        

    def _evaluate(self, metadata_configuration: typing.Dict) -> typing.Tuple[float, dict]:

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

        data = {
            'runtime': run,
            'pipeline': pipeline,
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics
        }
        # Use first metric from validation
        return validation_metrics[0]['value'], data


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
