import copy
import logging
import sys
import time
import typing
from multiprocessing import current_process
from pprint import pprint
from warnings import warn

from d3m.container.dataset import Dataset
from d3m.exceptions import NotSupportedError
from d3m.metadata.base import Metadata
from d3m.metadata.problem import PerformanceMetric
from dsbox.combinatorial_search.cache import PrimitivesCache
from dsbox.combinatorial_search.search_utils import get_target_columns
from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.pipeline.utils import larger_is_better
from dsbox.schema.problem import OptimizationType
from dsbox.schema.problem import optimization_type
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.configuration_space import ConfigurationSpace
from dsbox.template.template import DSBoxTemplate

# from dsbox.template.pipeline_utilities import pipe2str

T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class ConfigurationSpaceBaseSearch(typing.Generic[T]):
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

    def __init__(self, template: DSBoxTemplate,
                 configuration_space: ConfigurationSpace[T],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 performance_metrics: typing.List[typing.Dict], output_directory: str,
                 log_dir: str, ) -> None:

        self.template = template

        self.configuration_space = configuration_space
        # self.dimension_ordering = configuration_space_list.get_dimension_search_ordering()

        self.problem = problem
        self.train_dataset1 = train_dataset1
        self.train_dataset2 = train_dataset2
        self.test_dataset1 = test_dataset1
        self.test_dataset2 = test_dataset2
        self.all_dataset = all_dataset

        self.performance_metrics = list(map(
            lambda d: {'metric': d['metric'].unparse(), 'params': d['params']},
            performance_metrics
        ))

        self.output_directory = output_directory
        self.log_dir = log_dir

        self.minimize = optimization_type(performance_metrics[0]['metric']) == \
                        OptimizationType.MINIMIZE

        # TODO These variables have not been used at all
        self.classification_metric = ('accuracy', 'precision', 'normalizedMutualInformation',
                                      'recall', 'f1', 'f1Micro', 'f1Macro', 'rocAuc', 'rocAucMicro',
                                      'rocAucMacro')
        self.regression_metric = ('meanSquaredError', 'rootMeanSquaredError',
                                  'rootMeanSquaredErrorAvg', 'meanAbsoluteError', 'rSquared',
                                  'jaccardSimilarityScore', 'precisionAtTopK', 'objectDetectionAP')

        self.quick_mode = False
        self.testing_mode = 0  # set default to not use cross validation mode
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

    def dummy_evaluate(self, ) -> None:
        """
        This method is only used to import tensorflow before running the parallel jobs
        Args:
            configuration:

        Returns:

        """
        configuration: ConfigurationPoint[PrimitiveDescription] = \
            self.configuration_space.get_first_assignment()

        pipeline = self.template.to_pipeline(configuration)
        return pipeline

    def evaluate_pipeline(self, args) -> typing.Dict:
        """
        Evaluate at configuration point.
        Note: This methods will modify the configuration point, by updating its data field.
        """
        configuration: ConfigurationPoint[PrimitiveDescription] = dict(args[0])
        cache: PrimitivesCache = args[1]
        dump2disk = args[2] if len(args) == 3 else True
        print("[INFO] Worker started, id:", current_process(), ",", dump2disk)

        evaluation_result = self._evaluate(configuration, cache, dump2disk)
        # try:
        #     evaluation_result = self._evaluate(configuration, cache, dump2disk)
        # except:
        #     traceback.print_exc()
        #     return None
        # configuration.data.update(new_data)
        return evaluation_result

    def _evaluate(self,
                  configuration: ConfigurationPoint,
                  cache: PrimitivesCache,
                  dump2disk: bool = True) -> typing.Dict:

        start_time = time.time()
        pipeline = self.template.to_pipeline(configuration)
        # Todo: update ResourceManager to run pipeline:  ResourceManager.add_pipeline(pipeline)

        # if in cross validation mode
        if self.testing_mode == 1:
            repeat_times = int(self.validation_config['cross_validation'])
            print("[INFO] Will use cross validation( n =", repeat_times,
                  ") to choose best primitives.")
            # start training and testing
            fitted_pipeline = FittedPipeline(pipeline, self.train_dataset1.metadata.query(())['id'],
                                             log_dir=self.log_dir,
                                             metric_descriptions=self.performance_metrics)
            fitted_pipeline.fit(cache=cache, inputs=[self.train_dataset1])
            # fitted_pipeline.fit(inputs=[self.train_dataset1])
            training_ground_truth = get_target_columns(self.train_dataset1, self.problem)
            training_prediction = fitted_pipeline.get_fit_step_output(
                self.template.get_output_step_number())

            training_metrics, test_metrics = self._calculate_score(
                training_ground_truth, training_prediction, None, None)

            # copy the cross validation score here to test_metrics for return
            test_metrics = copy.deepcopy(
                training_metrics)  # fitted_pipeline.get_cross_validation_metrics()
            # generate a test matrics results with score = worst value
            if larger_is_better(training_metrics):
                test_metrics[0]["value"] = 0
            else:
                test_metrics[0]["value"] = sys.float_info.max
            print("[INFO] Testing finish.!!!")

        # if in normal testing mode(including default testing mode with train/test one time each)
        else:
            if self.testing_mode == 2:
                repeat_times = int(self.validation_config['test_validation'])
            else:
                repeat_times = 1
            print("[INFO] Will use normal train-test mode ( n =", repeat_times,
                  ") to choose best primitives.")
            training_metrics = []
            test_metrics = []

            for each_repeat in range(repeat_times):
                # start training and testing
                fitted_pipeline = FittedPipeline(pipeline,
                                                 self.train_dataset2[each_repeat].metadata.query(
                                                     ())['id'],
                                                 log_dir=self.log_dir,
                                                 metric_descriptions=self.performance_metrics)

                fitted_pipeline.fit(cache=cache, inputs=[self.train_dataset2[each_repeat]])
                # fitted_pipeline.fit(inputs=[self.train_dataset2[each_repeat]])
                training_ground_truth = get_target_columns(self.train_dataset2[each_repeat],
                                                           self.problem)
                training_prediction = fitted_pipeline.get_fit_step_output(
                    self.template.get_output_step_number())
                # only do test if the test_dataset exist
                if self.test_dataset2[each_repeat] is not None:
                    results = fitted_pipeline.produce(inputs=[self.test_dataset2[each_repeat]])
                    test_ground_truth = get_target_columns(self.test_dataset2[each_repeat],
                                                           self.problem)
                    # Note: results == test_prediction
                    test_prediction = fitted_pipeline.get_produce_step_output(
                        self.template.get_output_step_number())
                else:
                    test_ground_truth = None
                    test_prediction = None
                training_metrics_each, test_metrics_each = self._calculate_score(
                    training_ground_truth, training_prediction, test_ground_truth, test_prediction)

                # if no test_dataset exist, we need to give it with a default value
                if len(test_metrics_each) == 0:
                    test_metrics_each = copy.deepcopy(training_metrics_each)
                    if larger_is_better(training_metrics):
                        test_metrics_each[0]["value"] = 0
                    else:
                        test_metrics_each[0]["value"] = sys.float_info.max

                training_metrics.append(training_metrics_each)
                test_metrics.append(test_metrics_each)
            # sample format of the output
            # [{'metric': 'f1Macro', 'value': 0.48418535913661614, 'values': [0.4841025641025641,
            #  0.4841025641025641, 0.4843509492047203]]
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
            else:
                if type(test_metrics[0]) is list:
                    test_metrics = test_metrics[0]
                    training_metrics = training_metrics[0]
        # END evaluation part

        # Save results
        if self.test_dataset1 is None:
            print("The dataset no need to split of split failed, will not train again.")
            fitted_pipeline2 = fitted_pipeline
            # set the metric for calculating the rank
            fitted_pipeline2.set_metric(training_metrics[0])

            data = {
                'fitted_pipeline': fitted_pipeline2,
                'training_metrics': training_metrics,
                'cross_validation_metrics': fitted_pipeline2.get_cross_validation_metrics(),
                'test_metrics': training_metrics,
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
            }
            fitted_pipeline.auxiliary = dict(data)

            print("!!!! No test_dataset1")
            pprint(data)
            print("!!!!")

            if self.output_directory is not None and dump2disk:
                fitted_pipeline2.save(self.output_directory)

            #     _logger.info("Test pickled pipeline. id: {}".format(fitted_pipeline2.id))
            #     self.test_pickled_pipeline(folder_loc=self.output_directory,
            #                                pipeline_id=fitted_pipeline2.id,
            #                                test_dataset=self.train_dataset2[0],
            #                                test_metrics=training_metrics,
            #                                test_ground_truth=get_target_columns(self.train_dataset2[each_repeat], self.problem))
        else:
            if self.quick_mode or self.train_dataset1 is None:
                print("[INFO] Now in quick mode, will skip training with train_dataset1")
                # if in quick mode, we did not fit the model with dataset_train1 again
                # just generate the predictions on dataset_test1 directly and get the rank
                fitted_pipeline2 = fitted_pipeline
            else:
                print("[INFO] Now in normal mode, will add extra train with train_dataset1")
                # pipeline = self.template.to_pipeline(configuration)
                # otherwise train again with dataset_train1 and get the rank
                fitted_pipeline2 = FittedPipeline(pipeline,
                                                  self.train_dataset1.metadata.query(())['id'],
                                                  log_dir=self.log_dir,
                                                  metric_descriptions=self.performance_metrics)
                # retrain and compute ranking/metric using self.train_dataset
                # fitted_pipeline2.fit(inputs = [self.train_dataset1])
                fitted_pipeline2.fit(cache=cache, inputs=[self.train_dataset1])

            fitted_pipeline2.produce(inputs=[self.test_dataset1])
            test_ground_truth = get_target_columns(self.test_dataset1, self.problem)
            # Note: results == test_prediction
            test_prediction = fitted_pipeline2.get_produce_step_output(
                self.template.get_output_step_number())

            training_metrics2, test_metrics2 = self._calculate_score(
                None, None, test_ground_truth, test_prediction)

            # set the metric for calculating the rank
            fitted_pipeline2.set_metric(test_metrics2[0])

            # finally, fit the model with all data and save it
            print("[INFO] Now are training the pipeline with all dataset and saving the pipeline.")
            fitted_pipeline2.fit(cache=cache, inputs=[self.all_dataset])
            # fitted_pipeline2.fit(inputs = [self.all_dataset])

            data = {
                'fitted_pipeline': fitted_pipeline2,
                'training_metrics': training_metrics,
                'cross_validation_metrics': fitted_pipeline2.get_cross_validation_metrics(),
                'test_metrics': test_metrics2,
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
            }
            fitted_pipeline.auxiliary = dict(data)

            print("!!!!!! TEST_DATASET1")
            pprint(data)
            print("!!!!")

            if self.output_directory is not None and dump2disk:
                fitted_pipeline2.save(self.output_directory)

                # _ = fitted_pipeline2.produce(inputs=[self.test_dataset1])
                # test_prediction3 = fitted_pipeline2.get_produce_step_output(self.template.get_output_step_number())
                # _, test_metrics3 = self._calculate_score(None, None, test_ground_truth, test_prediction3)

            #     _logger.info("Test pickled pipeline. id: {}".format(fitted_pipeline2.id))
            #     self.test_pickled_pipeline(folder_loc=self.output_directory,
            #                                pipeline_id=fitted_pipeline2.id,
            #                                test_dataset=self.test_dataset1,
            #                                test_metrics=test_metrics3,
            #                                test_ground_truth=test_ground_truth)

        # still return the original fitted_pipeline with relation to train_dataset1
        return data

    def _calculate_score(self, training_ground_truth, training_prediction, test_ground_truth,
                         test_prediction):
        '''
            Ineer function used to calculate the score of the training and testing results based
            on given matrics
        '''
        training_metrics = []
        test_metrics = []
        for metric_description in self.performance_metrics:
            metricDesc = PerformanceMetric.parse(metric_description['metric'])
            metric: typing.Callable = metricDesc.get_function()
            params: typing.Dict = metric_description['params']
            regression_mode = metric_description["metric"] in self.regression_metric
            try:
                # generate the metrics for training results
                if training_ground_truth is not None and training_prediction is not None:  # if
                    # training data exist
                    if regression_mode:
                        training_metrics.append({
                            'metric': metric_description['metric'],
                            'value': metric(
                                training_ground_truth.iloc[:, -1].astype(float),
                                training_prediction.iloc[:, -1].astype(float),
                                **params
                            )
                        })
                    else:
                        if training_ground_truth is not None and training_prediction is not None:
                            # if training data exist
                            training_metrics.append({
                                'metric': metric_description['metric'],
                                'value': metric(
                                    training_ground_truth.iloc[:, -1].astype(str),
                                    training_prediction.iloc[:, -1].astype(str),
                                    **params
                                )
                            })
                # generate the metrics for testing results
                if test_ground_truth is not None and test_prediction is not None:  # if testing
                    # data exist
                    # if the test_ground_truth do not have results
                    if regression_mode:
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

        # if len(training_metrics) == 1:
        #     training_metrics = training_metrics[0]
        # el
        if len(training_metrics) > 1:
            print("[WARN] More than one training metrics found in one evaluation.")
        # if len(test_metrics) == 1:
        #     test_metrics = test_metrics[0]
        # el
        if len(test_metrics) > 1:
            print("[WARN] More than one test metrics found in one evaluation.")

        # return the training and test metrics
        return (training_metrics, test_metrics)

    def test_pickled_pipeline(self,
                              folder_loc: str,
                              pipeline_id: str,
                              test_dataset: Dataset,
                              test_metrics: typing.List,
                              test_ground_truth) -> None:
        fitted_pipeline, run = FittedPipeline.load(folder_loc=folder_loc, pipeline_id=pipeline_id,
                                                   log_dir=self.log_dir)
        results = fitted_pipeline.produce(inputs=[test_dataset])

        pipeline_prediction = fitted_pipeline.get_produce_step_output(
            self.template.get_output_step_number())

        test_pipeline_metrics2 = self._calculate_score(None, None, test_ground_truth,
                                                       pipeline_prediction)
        test_pipeline_metrics = list()
        for metric_description in self.performance_metrics:
            metricDesc = PerformanceMetric.parse(metric_description['metric'])
            metric: typing.Callable = metricDesc.get_function()
            params: typing.Dict = metric_description['params']

            try:
                if metric_description["metric"] in self.regression_metric:
                    # if the test_ground_truth do not have results
                    if test_ground_truth.iloc[0, -1] == '':
                        test_ground_truth.iloc[:, -1] = 0
                    test_pipeline_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(
                            test_ground_truth.iloc[:, -1].astype(float),
                            pipeline_prediction.iloc[:, -1].astype(float),
                            **params
                        )
                    })
                else:
                    test_pipeline_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(
                            test_ground_truth.iloc[:, -1].astype(str),
                            pipeline_prediction.iloc[:, -1].astype(str),
                            **params
                        )
                    })
            except:
                raise NotSupportedError(
                    '[ERROR] metric calculation failed in test pickled pipeline')

        print('=== original')
        pprint(test_metrics)
        print('=== test')
        print(test_pipeline_metrics)
        print('=== test2')
        print(test_pipeline_metrics2)

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
                "Test pickled pipeline mismatch. 'id': '%(id)s', 'test__metric': '%("
                "test__metric)s', 'pickled_pipeline__metric': '%(pickled_pipeline__metric)s'.",
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


