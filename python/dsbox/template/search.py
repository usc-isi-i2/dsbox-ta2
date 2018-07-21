import bisect
import operator
import os
import random
import traceback
import typing
import logging
import time
import copy
import pprint
import pathos.pools as pp

from warnings import warn

from multiprocessing import Pool, current_process, Manager
from itertools import zip_longest
from pprint import pprint

from d3m.exceptions import NotSupportedError
from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from d3m.metadata.base import ALL_ELEMENTS
from d3m.metadata.problem import PerformanceMetric
from d3m.primitive_interfaces.base import PrimitiveBaseMeta
from d3m.metadata.problem import TaskType

from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.pipeline.utils import larger_is_better
from dsbox.schema.problem import optimization_type
from dsbox.schema.problem import OptimizationType

from .template import DSBoxTemplate
from .template import HyperparamDirective

from .configuration_space import DimensionName
from .configuration_space import ConfigurationPoint
from .configuration_space import ConfigurationSpace
from .configuration_space import SimpleConfigurationSpace

from .pipeline_utilities import pipe2str

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

T = typing.TypeVar("T")
_logger = logging.getLogger(__name__)


def split_dataset(dataset, problem_info, problem_loc=None, *, random_state=42, test_size=0.2, n_splits = 1):
    '''
    Split dataset into training and test
    '''
    task_type = problem_info["task_type"]#['problem']['task_type'].name  # 'classification' 'regression'
    res_id = problem_info["res_id"]
    target_index = problem_info["target_index"]

    # for i in range(len(problem['inputs'])):
    #     if 'targets' in problem['inputs'][i]:
    #         break
    # task_type : str = problem['problem']['task_type'].name  # 'classification' 'regression'
    # res_id = problem['inputs'][i]['targets'][0]['resource_id']
    # target_index = problem['inputs'][i]['targets'][0]['column_index']

    def generate_split_data(dataset,res_id):
        train = dataset[res_id].iloc[train_index,:]
        test = dataset[res_id].iloc[test_index,:]
        # Generate training dataset
        train_dataset = copy.copy(dataset)
        train_dataset[res_id] = train
        meta = dict(train_dataset.metadata.query((res_id,)))
        dimension = dict(meta['dimension'])
        meta['dimension'] = dimension
        dimension['length'] = train.shape[0]
        #print(meta)
        train_dataset.metadata = train_dataset.metadata.update((res_id,), meta)
        #pprint(dict(train_dataset.metadata.query((res_id,))))
        # Generate testing dataset
        test_dataset = copy.copy(dataset)
        test_dataset[res_id] = test
        meta = dict(test_dataset.metadata.query((res_id,)))
        dimension = dict(meta['dimension'])
        meta['dimension'] = dimension
        dimension['length'] = test.shape[0]
        #print(meta)
        test_dataset.metadata = test_dataset.metadata.update((res_id,), meta)
        #pprint(dict(test_dataset.metadata.query((res_id,))))
        return (train_dataset,test_dataset)

    train_return = []
    test_return = []
    if task_type == 'CLASSIFICATION':
        # Use stratified sample to split the dataset
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        sss.get_n_splits(dataset[res_id], dataset[res_id].iloc[:, target_index])
        for train_index, test_index in sss.split(dataset[res_id], dataset[res_id].iloc[:, target_index]):
            train_dataset,test_dataset = generate_split_data(dataset,res_id)
            train_return.append(train_dataset)
            test_return.append(test_dataset)
    else:
        # Use random split
        if not task_type == "REGRESSION":
            print('USING Random Split to split task type: {}'.format(task_type))
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        ss.get_n_splits(dataset[res_id])
        for train_index, test_index in ss.split(dataset[res_id]):
            train_dataset,test_dataset = generate_split_data(dataset,res_id)
            train_return.append(train_dataset)
            test_return.append(test_dataset)

    return (train_return, test_return)

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
                        candidate_value: float = None, max_per_dimension=50, cache = None):
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

        # setup the output cache
        if not cache:
            local_cache = True
            manager = Manager()
            cache = manager.dict()

        # initialize the simulation counter
        sim_counter = 0
        start_time = time.clock()

        # we first need the baseline for searching the conf_space. For this
        # purpose we initially use first configuration and evaluate it on the
        #  dataset. In case that failed we repeat the sampling process one
        # more time to guarantee robustness on error reporting

        candidate, candidate_value = self.setup_initial_candidate(candidate_in, cache)
        sim_counter = 1

        # generate an executable pipeline with random steps from conf. space.
        # The actual searching process starts here.
        for dimension in self.dimension_ordering:
            # get all possible choices for the step, as specified in
            # configuration space
            choices: typing.List[T] = self.configuration_space.get_values(dimension)

            # TODO this is just a hack
             # if only have one candidate primitive for this step, no need to search
            if len(choices) == 1:
                continue
            # print("[INFO] choices:", choices, ", in step:", dimension)
            assert 1 < len(choices), \
                f'Step {dimension} has not primitive choices!'

            # the weights are assigned by template designer
            weights = [self.configuration_space.get_weight(dimension, x) for x in choices]

            # generate the candidates choices list
            selected = random_choices_without_replacement(choices, weights, max_per_dimension)

            #test_values = []
            score_values =[]
            sucessful_candidates = []
            import pdb
            pdb.set_trace()

            # No need to evaluate if value is already known
            if candidate_value is not None and candidate[dimension] in selected:
                selected.remove(candidate[dimension])
                # also add this candidate to the succesfully_candidates to make comparisons easier
                sucessful_candidates.append(candidate)
                score_values.append(candidate_value)

            # all the new possible pipelines are generated in new_candidates
            new_candidates: typing.List[ConfigurationPoint] = []
            for value in selected:
                # transfer the pipeline to dictionary type so that we can change detail steps
                new = dict(candidate)
                # replace the traget step
                new[dimension] = value
                # regenerate the pipeline
                candidate_ = self.configuration_space.get_point(new)
                new_candidates.append(candidate_)

            best_index = -1
            print('*' * 100)
            print("[INFO] Running Pool for step", dimension,", fork_num:", len(new_candidates))
            sim_counter += len(new_candidates)
            # run all candidate pipelines in multi-processing mode
            try:
                with pp.ProcessPool(self.num_workers) as p:
                    results = p.map(
                        self.evaluate,
                        map(lambda c: (c, cache), new_candidates)
                    )
                for res, x in zip(results, new_candidates):
                    if not res:
                        print('[ERROR] candidate failed:')
                        pprint(x)
                        print("-"*10)
                        continue
                    if len(res['cross_validation_metrics']) > 0:
                        cross_validation_mode = True
                        score_values.append(res['cross_validation_metrics'][0]['value'])
                    else:
                        score_values.append(res['test_metrics'][0]['value'])
                        cross_validation_mode = False
                    # pipeline = self.template.to_pipeline(x)
                    # res['pipeline'] = pipeline
                    res['fitted_pipeline'] = res['fitted_pipeline']
                    x.data.update(res)
                    sucessful_candidates.append(x)
            except:
                traceback.print_exc()

            import pdb
            pdb.set_trace()

            # If all candidates failed, only the initial one in the score_values
            if len(score_values) == 1:
                print("[INFO] No new Candidate worked in this step!")
                if not candidate:
                    print("[ERROR] The template did not return any valid pipelines!")
                    return (None, None)
                else:
                    continue
            # Find best candidate
           #best_cv_index = 0 # initialize best_cv_index
            if self.minimize:
                best_index = score_values.index(min(score_values))
            else:
                best_index = score_values.index(max(score_values))

            # # for conditions that no test dataset given or in the new training mode
            # if sum(test_values) == 0 and cross_validation_values:
            #     best_index = best_cv_index
            if cross_validation_mode:
                print("[INFO] Best index:", best_index, " --> CV matrix score:", score_values[best_index])
            else:
                print("[INFO] Best index:", best_index, " --> Test matrix score:", score_values[best_index])

            # put the best candidate pipeline and results to candidate
            candidate = sucessful_candidates[best_index]
            candidate_value = score_values[best_index]

        # END FOR

        # shutdown the cache manager
        if local_cache:
            manager.shutdown()

        # here we can get the details of pipelines from "candidate.data"
        assert "fitted_pipeline" in candidate.data, "parameters not added! last"
        # gather information for UCT

        # if 'test_values' in locals() and test_values:
        #     reward = abs(sum(test_values)) / len(test_values)
        # else:
        reward = candidate_value

        # TODO : come up with a better method for this
        if self.minimize:
            reward = -reward

        UCT_report = {
            'reward': reward,
            'time': time.clock() - start_time,
            'sim_count': sim_counter,
            'candidate': candidate,
            'best_val': candidate_value
        }
        # return (candidate, candidate_value)
        return UCT_report



    def setup_initial_candidate(self,
                                candidate: ConfigurationPoint[T],
                                cache: typing.Dict) -> \
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
            candidate = ConfigurationPoint(self.configuration_space, self.first_assignment())
        # first, then random, then another random
        for i in range(2):
            try:
                result = self.evaluate((candidate, cache))
                candidate.data.update(result)

                if 'cross_validation_metrics' in result and len(result['cross_validation_metrics']) > 0:
                    return (candidate, result['cross_validation_metrics'][0]['value'])
                else:
                    return (candidate, result['test_metrics'][0]['value'])
            except:
                traceback.print_exc()
                print("[ERROR] Initial Pipeline failed, Trying a random pipeline ...")
                pprint(candidate)
                print("-"*20)
                candidate = ConfigurationPoint(self.configuration_space,
                                               self.random_assignment())
        raise ValueError("Invalid initial candidate")
        # result = self.evaluate((candidate, cache))
        # candidate.data.update(result)
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
        # return (candidate, result['test_metrics'][0]['value'])



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
    primitive_index : typing.List[str]
        List of primitive python paths from d3m.index.search()
    train_dataset : Dataset
        The dataset to train pipeline
    test_dataset : Dataset
        The dataset to evaluate pipeline
    performance_metrics : typing.List[typing.Dict]
        Performance metrics from parse_problem_description()['problem']['performance_metrics']
    """

    def __init__(self, 
                 template: DSBoxTemplate,
                 config: typing.Dict,
                 problem_dict: typing.Dict,
                 configuration_space: ConfigurationSpace[PrimitiveDescription],
                 problem: Metadata,
                 all_dataset: Dataset,
                 performance_metrics: typing.List[typing.Dict],
                 output_directory: str,
                 log_dir: str,
                 num_workers: int = 0) -> None:

        # Use first metric from test
        minimize = optimization_type(performance_metrics[0]['metric']) == OptimizationType.MINIMIZE
        super().__init__(self.evaluate_pipeline, configuration_space, minimize)

        self.template = template
        self.config = config
        # self.configuration_space = configuration_space
        #self.primitive_index: typing.List[str] = primitive_index
        self.problem = problem
        self.problem_info = self._generate_problem_info(problem_dict)
        self.train_dataset1 = None
        self.test_dataset1 = None
        # self.train_dataset2 = None
        # self.test_dataset2 = None
        self.all_dataset = all_dataset

        self.performance_metrics = list(map(
            lambda d: {'metric': d['metric'].unparse(), 'params': d['params']},
            performance_metrics
        ))

        self.output_directory = output_directory
        self.log_dir = log_dir
        self.num_workers = os.cpu_count() if num_workers==0 else num_workers

        # new searching method: first check whether we will do corss validation or not
        self.quick_mode = False
        self.testing_mode = 0 # set default to not use cross validation mode
        # testing_mode = 0: normal testing mode with test only 1 time
        # testing_mode = 1: cross validation mode
        # testing_mode = 2: multiple testing mode with testing with random split data n times
        self.validation_config = None
        for each_step in template.template['steps']:
            if 'runtime' in each_step:
                self.validation_config = each_step['runtime']
                if "cross_validation" in each_step['runtime']:
                    self.testing_mode = 1
                else:
                    self.testing_mode = 2
        # now we split dataset here to keep only one spliting function
        self.train_dataset1, self.test_dataset1 = split_dataset(dataset = self.all_dataset, 
            problem_info = self.problem_info, problem_loc = self.config['problem_schema'])
        # here we only split one times, so no need to use list to include the dataset
        self.train_dataset1 = self.train_dataset1[0]
        self.test_dataset1 = self.test_dataset1[0]
        # if not set(self.template.template_nodes.keys()) <= set(configuration_space.get_dimensions()):
        #     raise exceptions.InvalidArgumentValueError(
        #         "Not all template steps are in configuration space: {}".format(self.template.template_nodes.keys()))

    def _generate_problem_info(self,problem):
        problem_info = {}
        for i in range(len(problem['inputs'])):
            if 'targets' in problem['inputs'][i]:
                break
        problem_info["task_type"] = problem['problem']['task_type'].name  # 'classification' 'regression'
        problem_info["res_id"] = problem['inputs'][i]['targets'][0]['resource_id']
        problem_info["target_index"] = problem['inputs'][i]['targets'][0]['column_index']
        return problem_info

    def evaluate_pipeline(self, args) -> typing.Dict:
        """
        Evaluate at configuration point.
        Note: This methods will modify the configuration point, by updating its data field.
        """
        configuration: ConfigurationPoint[PrimitiveDescription] = args[0]
        cache: typing.Dict = args[1]
        print("[INFO] Worker started, id:", current_process())
        try:
            evaluation_result = self._evaluate(configuration, cache)
        except:
            traceback.print_exc()
            return None
        # configuration.data.update(new_data)
        return evaluation_result

    def _evaluate(self,
                  configuration: ConfigurationPoint,
                  cache: typing.Dict) -> typing.Dict:
        pipeline = self.template.to_pipeline(configuration)
        # Todo: update ResourceManager to run pipeline:  ResourceManager.add_pipeline(pipeline)
        # if in cross validation mode
        if self.testing_mode == 1:
            repeat_times = int(self.validation_config['cross_validation'])
            print("[INFO] Will use cross validation( n =", repeat_times,") to choose best primitives.")
            # start training and testing
            fitted_pipeline = FittedPipeline(pipeline, self.train_dataset1.metadata.query(())['id'], 
                log_dir=self.log_dir, metric_descriptions=self.performance_metrics)
            #fitted_pipeline.fit(cache=cache, inputs=[self.train_dataset1])
            fitted_pipeline.fit(inputs=[self.train_dataset1])
            training_ground_truth = get_target_columns(self.train_dataset1,self.problem)
            training_prediction = fitted_pipeline.get_fit_step_output(
            self.template.get_output_step_number())

            training_metrics, test_metrics = self._calculate_score(
                    training_ground_truth,training_prediction,None,None)

            # copy the cross validation score here to test_metrics for return
            test_metrics = fitted_pipeline.get_cross_validation_metrics()
            print("[INFO] Testing finish.!!!")
        else:
            if self.testing_mode == 2:
                repeat_times = int(self.validation_config['test_validation'])
            else:
                repeat_times = 1
            print("[INFO] Will use normal train-test mode ( n =", repeat_times,") to choose best primitives.")
            training_metrics = []
            test_metrics = []
            # make n times of different spliting results
            # train_dataset2, test_dataset2 = split_dataset(dataset = self.train_dataset1, 
            #     problem = self.problem_info, problem_loc = self.config['problem_schema'],
            #     test_size = 0.1, n_splits = repeat_times)

            for each_repeat in range(repeat_times):
                # start training and testing
                fitted_pipeline = FittedPipeline(pipeline, train_dataset2[each_repeat].metadata.query(())['id'], 
                    log_dir=self.log_dir, metric_descriptions=self.performance_metrics)

                #fitted_pipeline.fit(cache=cache, inputs=[train_dataset2[each_repeat]])
                fitted_pipeline.fit(inputs=[train_dataset2[each_repeat]])
                training_ground_truth = get_target_columns(train_dataset2[each_repeat],self.problem)
                training_prediction = fitted_pipeline.get_fit_step_output(
                    self.template.get_output_step_number())
                results = fitted_pipeline.produce(inputs=[test_dataset2[each_repeat]])
                test_ground_truth = get_target_columns(test_dataset2[each_repeat],self.problem)

                # Note: results == test_prediction
                test_prediction = fitted_pipeline.get_produce_step_output(self.template.get_output_step_number())

                training_metrics_each, test_metrics_each = self._calculate_score(
                    training_ground_truth,training_prediction,test_ground_truth,test_prediction)
                training_metrics.append(training_metrics_each)
                test_metrics.append(test_metrics_each)
            # sample format of the output
            #[{'metric': 'f1Macro', 'value': 0.48418535913661614, 'values': [0.4841025641025641, 0.4841025641025641, 0.4843509492047203]]
            # modify the test_metrics and training_metrics format to fit the requirements
            print("[INFO] Testing finish.!!!")
            if len(training_metrics) > 1:
                training_value_list = []
                test_value_list = []
                for each in training_metrics:
                    training_value_list.append(each['value'])
                    test_value_list.append(each['value'])
                # training_metrics part
                training_metrics_new = training_metrics[0]
                training_metrics_new['value'] = sum(training_value_list) / len(training_value_list)
                training_metrics_new['values'] = training_value_list
                training_metrics = [training_metrics_new]
                # test_metrics part
                test_metrics_new = test_metrics[0]
                test_metrics_new['value'] = sum(test_value_list) / len(test_value_list)
                test_metrics_new['values'] = test_value_list
                test_metrics = [test_metrics_new]

        # Save results
        if self.output_directory is not None:
            data = {
                'fitted_pipeline': fitted_pipeline,
                'training_metrics': training_metrics,
                'cross_validation_metrics': fitted_pipeline.get_cross_validation_metrics(),
                'test_metrics': test_metrics
            }

            if self.quick_mode:
                print("[INFO] Now in quick mode, will skip training with train_dataset1")
                # if in quick mode, we did not fit the model with dataset_train1 again
                # just generate the predictions on dataset_test1 directly and get the rank
                fitted_pipeline2 = fitted_pipeline
            else:
                print("[INFO] Now in normal mode, will add extra train with train_dataset1")
                # otherwise train again with dataset_train1 and get the rank
                fitted_pipeline2 = FittedPipeline(pipeline, self.train_dataset1.metadata.query(())['id'], 
                    log_dir=self.log_dir, metric_descriptions=self.performance_metrics)
                # retrain and compute ranking/metric using self.train_dataset
                fitted_pipeline2.fit(inputs = [self.train_dataset1])
                #fitted_pipeline2.fit(cache=cache, inputs = [self.train_dataset1])

            fitted_pipeline2.produce(inputs = [self.test_dataset1])
            test_ground_truth = get_target_columns(self.test_dataset1,self.problem)
            # Note: results == test_prediction
            test_prediction = fitted_pipeline2.get_produce_step_output(self.template.get_output_step_number())

            training_metrics, test_metrics = self._calculate_score(
                None, None, test_ground_truth, test_prediction)

            # set the metric for calculating the rank
            fitted_pipeline2.set_metric(test_metrics)

            # finally, fit the model with all data and save it
            print("[INFO] Now are training the pipeline with all dataset and saving the pipeline.")
            #fitted_pipeline2.fit(cache=cache, inputs = [self.all_dataset])
            fitted_pipeline2.fit(inputs = [self.all_dataset])
            fitted_pipeline2.save(self.output_directory)
            # _logger.info("Test pickled pipeline. id: {}".format(fitted_pipeline.id))
            # self.test_pickled_pipeline(folder_loc=self.output_directory,
            #                                pipeline_id=fitted_pipeline.id,
            #                                test_metrics=test_metrics,
            #                                test_ground_truth=test_ground_truth)

        # still return the original fitted_pipeline with relation to train_dataset1
        return data

    def _calculate_score(self, training_ground_truth, training_prediction, test_ground_truth, test_prediction):
        '''
            Ineer function used to calculate the score of the training and testing results based on given matrics
        '''
        training_metrics = []
        test_metrics = []
        for metric_description in self.performance_metrics:
            metricDesc = PerformanceMetric.parse(metric_description['metric'])
            metric: typing.Callable = metricDesc.get_function()
            params: typing.Dict = metric_description['params']
            try:                
                if 'regression' in self.problem.query(())['about']['taskType']:
                    if training_ground_truth is not None and training_prediction is not None: # if training data exist
                        training_metrics.append({
                            'metric': metric_description['metric'],
                            'value': metric(
                                training_ground_truth.iloc[:, -1].astype(float),
                                training_prediction.iloc[:, -1].astype(float),
                                **params
                            )
                        })

                    if test_ground_truth is not None and test_prediction is not None: # if testing data exist
                        # if the test_ground_truth do not have results
                        if test_ground_truth.iloc[0, -1] == '':
                            test_ground_truth.iloc[:, -1] = 0
                        test_metrics.append({
                            'metric': metric_description['metric'],
                            'value': metric(
                                test_ground_truth.iloc[:, -1].astype(float),
                                test_prediction.iloc[:, -1].astype(float),
                                **params
                            )
                        })
                else:
                    if training_ground_truth is not None and training_prediction is not None: # if training data exist
                        training_metrics.append({
                            'metric': metric_description['metric'],
                            'value': metric(
                                training_ground_truth.iloc[:, -1].astype(str),
                                training_prediction.iloc[:, -1].astype(str),
                                **params
                            )
                        })
                    if test_ground_truth is not None and test_prediction is not None: # if testing data exist
                        test_metrics.append({
                            'metric': metric_description['metric'],
                            'value': metric(
                                test_ground_truth.iloc[:, -1].astype(str),
                                test_prediction.iloc[:, -1].astype(str),
                                **params
                            )
                        })
            except:
                raise NotSupportedError('[ERROR] metric calculation failed')
        # END for loop

        if len(training_metrics) == 1:
            training_metrics = training_metrics[0]
        elif len(training_metrics) > 1:
            print("[WARN] More than one training metrics found in one evaluation.")
        if len(test_metrics) == 1:
            test_metrics = test_metrics[0]
        elif len(test_metrics) > 1:
            print("[WARN] More than one test metrics found in one evaluation.")
        
        # return the training and test metrics
        return (training_metrics, test_metrics)

    def test_pickled_pipeline(self,
                              folder_loc: str,
                              pipeline_id: str,
                              test_metrics: typing.List,
                              test_ground_truth) -> None:

        fitted_pipeline, run = FittedPipeline.load(folder_loc=folder_loc, pipeline_id=pipeline_id, log_dir=self.log_dir)
        results = fitted_pipeline.produce(inputs=[self.test_dataset])
        pipeline_pridiction = fitted_pipeline.get_produce_step_output(self.template.get_output_step_number())

        test_pipeline_metrics = list()
        for metric_description in self.performance_metrics:
            metricDesc = PerformanceMetric.parse(metric_description['metric'])
            metric: typing.Callable = metricDesc.get_function()
            params: typing.Dict = metric_description['params']

            try:
                if 'regression' in self.problem.query(())['about']['taskType']:
                    # if the test_ground_truth do not have results
                    if test_ground_truth.iloc[0, -1] == '':
                        test_ground_truth.iloc[:, -1] = 0
                    test_pipeline_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(
                            test_ground_truth.iloc[:, -1].astype(float),
                            pipeline_pridiction.iloc[:, -1].astype(float),
                            **params
                        )
                    })
                else:
                    test_pipeline_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(
                            test_ground_truth.iloc[:, -1].astype(str),
                            pipeline_pridiction.iloc[:, -1].astype(str),
                            **params
                        )
                    })
            except:
                raise NotSupportedError(
                    '[ERROR] metric calculation failed in test pickled pipeline')

        pairs = zip(test_metrics, test_pipeline_metrics)
        if any(x != y for x, y in pairs):
            warn("[WARN] Test pickled pipeline mismatch. id: {}".format(fitted_pipeline.id))
            print(
                {
                    'id': fitted_pipeline.id,
                    'test__metric': test_metrics,
                    'pickled_pipeline__metric': test_pipeline_metrics
                }
            )
            print("\n" * 5)
            _logger.warning(
                "Test pickled pipeline mismatch. 'id': '%(id)s', 'test__metric': '%(test__metric)s', 'pickled_pipeline__metric': '%(pickled_pipeline__metric)s'.",
                {
                    'id': fitted_pipeline.id,
                    'test__metric': test_metrics,
                    'pickled_pipeline__metric': test_pipeline_metrics
                },
            )
            print("\n" * 5)
        else:
            print("\n" * 5)
            print("Pickling succeeded")
            print("\n" * 5)


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
