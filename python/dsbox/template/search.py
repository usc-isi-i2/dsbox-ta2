import bisect
import operator
import random
import traceback
import typing

from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from d3m.metadata.base import ALL_ELEMENTS
from d3m.primitive_interfaces.base import PrimitiveBaseMeta
from dsbox.pipeline.fitted_pipeline import FittedPipeline


from dsbox.schema.problem import optimization_type, OptimizationType

from .template import HyperparamDirective, DSBoxTemplate

from .configuration_space import DimensionName, ConfigurationSpace, SimpleConfigurationSpace, ConfigurationPoint

from pprint import pprint
from .pipeline_utilities import pipe2str

from multiprocessing import Pool

T = typing.TypeVar("T")

def get_target_columns(dataset: 'Dataset', problem_doc_metadata: 'Metadata'):
    problem = problem_doc_metadata.query(())["inputs"]["data"]
    datameta = dataset.metadata
    target = problem[0]["targets"]
    resID = target[0]["resID"]
    colIndex = target[0]["colIndex"]
    datalength = datameta.query((resID, ALL_ELEMENTS,))["dimension"]['length']
    targetlist = []
    for v in range(datalength):
        types = datameta.query((resID, ALL_ELEMENTS, v))["semantic_types"]
        for t in types:
            if t == 'https://metadata.datadrivendiscovery.org/types/PrimaryKey':
                targetlist.append(v)
    targetlist.append(colIndex)
    targetcol = dataset[resID].iloc[:, targetlist]
    return targetcol


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

    def first_assignment(self) -> typing.Dict[DimensionName, T]:
        '''
        Assign the first value for each dimension
        '''
        assignment: typing.Dict[DimensionName, T] = {}
        for dimension in self.dimension_ordering:
            assignment[dimension] = self.configuration_space.get_values(dimension)[0]
        return assignment

    def get_dimension_length(self, kw: DimensionName) -> int:
        '''
        Return the length of the list a configuration point
        '''
        return len(self.configuration_space.get_values(kw))

    def generate_pipeline(self, configuration_space: ConfigurationSpace[T],
                          dimension: typing.List[DimensionName]):
        pass

    def search_one_iter(self, candidate_in: ConfigurationPoint[T] = None,
                        candidate_value: float = None, max_per_dimension=10):
        """
        Performs one iteration of dimensional search. During dimesional
        search our algorithm iterates through all 8 steps of pipeline as
        indicated in our configuration space and greedily optimizes the
        pipeline one step at a time.

        Parameters
        ----------
        candidate_in: ConfigurationPoint[T]
            Current best candidate
        candidate_value: float
            The valude for the current best candidate
        max_per_dimension: int
            Maximunum number of values to search per dimension
        """
        # we first need the baseline for searching the conf_space. For this
        # purpose we initially use first configuration and evaluate it on the
        #  dataset. In case that failed we repeat the sampling process one
        # more time to guarantee robustness on error reporting
        candidate, candidate_value = self.setup_initial_candidate(candidate_in)

        # generate an executable pipeline with random steps from conf. space.

        # The actual searching process starts here.
        for dimension in self.dimension_ordering:
            # get all possible choices for the step, as specified in
            # configuration space
            choices: typing.List[T] = self.configuration_space\
                                          .get_values(dimension)

            # TODO this is just a hack
            if len(choices) == 1:
                continue

            assert 0 < len(choices), \
                f'Step {dimension} has not primitive choices!'

            # the weights are assigned by template designer
            weights = [self.configuration_space.get_weight(
                dimension, x) for x in choices]

            selected = random_choices_without_replacement(
                choices, weights, max_per_dimension)

            # No need to evaluate if value is already known
            if candidate_value is not None and candidate[dimension] in selected:
                selected.remove(candidate[dimension])

            new_candidates: typing.List[ConfigurationPoint] = []
            for value in selected:
                new = dict(candidate)
                new[dimension] = value
                new_candidates.append(self.configuration_space.get_point(new))

            values = []
            sucessful_candidates = []

            try:
                with Pool(5) as p:
                    results = p.map(self.evaluate, new_candidates)

                for res, x in zip(results,new_candidates):
                    values.append(res[0])
                    sucessful_candidates.append(x)
            except:
            # print('Pipeline failed: ', x)
                traceback.print_exc()

            # for x in new_candidates:
            #     try:
            #         result = self.evaluate(x)
            #         values.append(result[0])
            #         sucessful_candidates.append(x)
            #         # print("[INFO] Results:")
            #         # pprint(result)
            #         # pprint(result[0])
            #     except:
            #         # print('Pipeline failed: ', x)
            #         traceback.print_exc()

            # All candidates failed!
            if len(values) == 0:
                print("[INFO] No Candidate worked!:",values)
                return (None, None)

            # Find best candidate
            if self.minimize:
                best_index = values.index(min(values))
            else:
                best_index = values.index(max(values))

            if candidate_value is None:
                candidate = sucessful_candidates[best_index]
                candidate_value = values[best_index]
            elif (self.minimize and values[best_index] < candidate_value) or (not self.minimize and values[best_index] > candidate_value):
                candidate = sucessful_candidates[best_index]
                candidate_value = values[best_index]
        # here we can get the details of pipelines from "candidate.data"

        return (candidate, candidate_value)



    def setup_initial_candidate(self, candidate: ConfigurationPoint[T]) -> \
            typing.Tuple[ConfigurationPoint[T], float]:
        """
        we first need the baseline for searching the conf_space. For this
        purpose we initially use first configuration and evaluate it on the
        dataset. In case that failed we repeat the sampling process one more
        time to guarantee robustness on error reporting

        Args:
            candidate: ConfigurationPoint[T]

        Returns:
            candidate, evaluate_value : ConfigurationPoint[T], float
        """
        if candidate is None:
            candidate = ConfigurationPoint(
                self.configuration_space, self.first_assignment())
        # first, then random, then another random
        # for i in range(2):
        #     try:
        #         result = self.evaluate(candidate)
        #         return (candidate, result[0])
        #     except:
        #         print("Pipeline failed")
        #         candidate = ConfigurationPoint(self.configuration_space,
        #                                        self.random_assignment())
        result = self.evaluate(candidate)
        # try:
        #     result = self.evaluate(candidate)
        # except:
        #     print("***************")
        #     print("Pipeline failed")
        #     candidate = ConfigurationPoint(self.configuration_space,
        #                                    self.random_assignment())
        #     try:
        #         result = self.evaluate(candidate)
        #     except:
        #         print("Pipeline failed")
        #         candidate = ConfigurationPoint(self.configuration_space,
        #                                        self.random_assignment())
        #         result = self.evaluate(candidate)
        return (candidate, result[0])



    def search(self, candidate: ConfigurationPoint[T] = None, candidate_value: float = None, num_iter=3, max_per_dimension=10):
        for i in range(num_iter):
            candidate, candidate_value = self.search_one_iter(candidate, candidate_value, max_per_dimension=max_per_dimension)
            if candidate is None:
                return (None, None)

        return (candidate, candidate_value)


# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)


class TemplateDimensionalSearch(DimensionalSearch[PrimitiveDescription]):
    """
    Use dimensional search to find best pipeline.

    Attributes
    ----------
    template : DSBoxTemplate
        The template pipeline to be fill in
    configuration_space : ConfigurationSpace[PrimitiveDescription]
        Configuration space where values are primitive python paths
    train_dataset : Dataset
        The dataset to train pipeline
    validation_dataset : Dataset
        The dataset to evaluate pipeline
    performance_metrics : typing.List[typing.Dict]
        Performance metrics from parse_problem_description()['problem']['performance_metrics']
    """

    def __init__(self, template: DSBoxTemplate,
                 configuration_space: ConfigurationSpace[PrimitiveDescription],
                 problem: Metadata,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 performance_metrics: typing.List[typing.Dict]) -> None:

        # Use first metric from validation

        minimize = optimization_type(performance_metrics[0]['metric']) == OptimizationType.MINIMIZE
        super().__init__(self.evaluate_pipeline, configuration_space, minimize)

        self.template: DSBoxTemplate = template
        # self.configuration_space = configuration_space
        self.problem = problem
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.performance_metrics = performance_metrics

        # if not set(self.template.template_nodes.keys()) <= set(configuration_space.get_dimensions()):
        #     raise exceptions.InvalidArgumentValueError(
        #         "Not all template steps are in configuration space: {}".format(self.template.template_nodes.keys()))

    def evaluate_pipeline(self, configuration: ConfigurationPoint[PrimitiveDescription]) -> typing.Tuple[float, dict]:
        """
        Evaluate at configuration point.
        Note: This methods will modify the configuration point, by updating its data field.
        """

        # convert PrimitiveDescription to primitive metadata
        # metadata_configuration: typing.Dict[DimensionName, PrimitiveMetadata] = {
        #     key: self.primitive_index[python_path].metadata.query() for key, python_path in configuration.items()}

        # value, new_data = self._evaluate(metadata_configuration)

        value, new_data = self._evaluate(configuration)
        configuration.data.update(new_data)
        return value, configuration.data

    def _evaluate(self, configuration: ConfigurationPoint) -> typing.Tuple[float, dict]:

        pipeline = self.template.to_pipeline(configuration)

        # Todo: update ResourceManager to run pipeline:  ResourceManager.add_pipeline(pipeline)
        fitted_pipeline = FittedPipeline(pipeline, dataset_id=self.train_dataset.metadata.query(())['id'])

        fitted_pipeline.fit(inputs=[self.train_dataset])
        training_ground_truth = get_target_columns(self.train_dataset, self.problem)
        training_prediction = fitted_pipeline.get_fit_step_output(self.template.get_output_step_number())

        print('*'*100)
        results = fitted_pipeline.produce(inputs=[self.validation_dataset])
        validation_ground_truth = get_target_columns(self.validation_dataset, self.problem)
        # Note: results == validation_prediction
        validation_prediction = fitted_pipeline.get_produce_step_output(self.template.get_output_step_number())

        training_metrics = []
        validation_metrics = []
        for metric_description in self.performance_metrics:
            metric: typing.Callable = metric_description['metric'].get_function()
            params: typing.Dict = metric_description['params']

            try:
                if 'regression' in self.problem.query(())['about']['taskType']:
                    training_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(training_ground_truth.iloc[:, -1].astype(float), training_prediction.iloc[:, -1].astype(float))
                    })
                    # if the validation_ground_truth do not have results
                    if validation_ground_truth.iloc[0, -1] == '':
                        validation_ground_truth.iloc[:, -1] = 0
                    validation_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(validation_ground_truth.iloc[:, -1].astype(float), validation_prediction.iloc[:, -1].astype(float))
                    })
                else:
                    training_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(training_ground_truth.iloc[:, -1].astype(str), training_prediction.iloc[:, -1].astype(str))
                    })
                    validation_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(validation_ground_truth.iloc[:, -1].astype(str), validation_prediction.iloc[:, -1].astype(str))
                    })
            except:
                import pdb
                pdb.set_trace()

        data = {
            'fitted_pipeline': fitted_pipeline,
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics
        }
        # Use first metric from validation
        print(pipe2str(pipeline))
        pprint(data)
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
