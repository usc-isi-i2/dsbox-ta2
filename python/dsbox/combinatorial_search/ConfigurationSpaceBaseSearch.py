import copy
import logging
import sys
import time
import typing
import enum
import collections
import frozendict
# import eventlet

from multiprocessing import current_process
from warnings import warn

from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
from d3m.base import utils as d3m_utils
from d3m.exceptions import NotSupportedError
from d3m.metadata.base import Metadata, ALL_ELEMENTS
from d3m.metadata.problem import PerformanceMetric, Problem, TaskType
from dsbox.JobManager.cache import PrimitivesCache
from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.schema import get_target_columns, larger_is_better
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.configuration_space import ConfigurationSpace
from dsbox.template.template import DSBoxTemplate
# from dsbox.template.utils import calculate_score, graph_problem_conversion, SpecialMetric
from dsbox.template.utils import score_prediction, graph_problem_conversion, SpecialMetric
from datamart_isi.entries import AUGMENTED_COLUMN_SEMANTIC_TYPE, Q_NODE_SEMANTIC_TYPE
T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class Mode(enum.IntEnum):
    CROSS_VALIDATION_MODE = 1
    TRAIN_TEST_MODE = 2


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

    TODO:
        1. break the evaluation method into multiple train-test method calls.
        2. Make the evaluation parametrized on the sort of evaluation mode
           ( ('partial', 'whole'), ('bootstrap', 'cross-validation'))
    """

    def __init__(self, template: DSBoxTemplate,
                 configuration_space: ConfigurationSpace[T],
                 problem: Problem, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 ensemble_tuning_dataset: Dataset,
                 performance_metrics: typing.List[typing.Dict], output_directory: str,
                 extra_primitive: typing.Set[str] = set(), *,
                 random_seed: int = 0
    ) -> None:

        self.template = template
        self.task_type = self.template.template["taskType"]
        self.random_seed = random_seed

        self.configuration_space = configuration_space
        # self.dimension_ordering = configuration_space_list.get_dimension_search_ordering()

        self.problem: Problem = problem
        self.train_dataset1 = train_dataset1
        self.train_dataset2 = train_dataset2
        self.test_dataset1 = test_dataset1
        self.test_dataset2 = test_dataset2
        self.all_dataset = all_dataset
        self.extra_primitive = extra_primitive
        if ensemble_tuning_dataset:
            self.do_ensemble_tuning = True
            self.ensemble_tuning_dataset = ensemble_tuning_dataset
        else:
            self.do_ensemble_tuning = False
            self.ensemble_tuning_dataset = None

        self.performance_metrics = performance_metrics

        self.output_directory = output_directory

        self.minimize = performance_metrics[0]['metric'].best_value() < performance_metrics[0]['metric'].worst_value()

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
                    _logger.debug("Will use cross validation(n = {}) to choose best primitives".format(int(self.validation_config['cross_validation'])))
                    _logger.info("Validation mode: Cross Validation")
                    # print("!!!!!@@#### CV mode!!!")
                    break
                else:
                    self.testing_mode = 2
                    _logger.debug("Will use test_dataset to choose best primitives")
                    _logger.info("Validation mode: normal")
                    # print("!!!!!@@#### normal mode!!!")
                    break

        # new searching method: first check whether we should train a second time with
        # dataset_train1
        self.go_quick_inputType = ["image", "audio", "video"]
        self.quick_mode = self._use_quick_mode_or_not()

    def _use_quick_mode_or_not(self) -> bool:
        """
        The function to determine whether to use quick mode or now
            Now it is hard coded
        Returns:
            use_quick mode?
        """
        for each_type in self.template.template['inputType']:
            if each_type in self.go_quick_inputType:
                return True
        return False

    # def dummy_evaluate(self, ) -> None:
    #     """
    #     This method is only used to import tensorflow before running the parallel jobs
    #     Args:
    #         configuration:
    #
    #     Returns:
    #
    #     """
    #     _logger.info("Dummy evaluation started")
    #     configuration: ConfigurationPoint[PrimitiveDescription] = \
    #         self.configuration_space.get_first_assignment()
    #
    #     pipeline = self.template.to_pipeline(configuration)
    #     return pipeline

    def evaluate_pipeline(self, args) -> typing.Dict:
        """
        Evaluate at configuration point.
        Note: This methods will modify the configuration point, by updating its data field.
        """

        configuration: ConfigurationPoint[PrimitiveDescription] = dict(args[0])
        cache: PrimitivesCache = args[1]
        dump2disk = args[2] if len(args) == 3 else True

        evaluation_result = None

        try:
            _logger.info(f"START Evaluation of template {self.template.template['name']} {hash(str(configuration))} in {current_process()}")

            evaluation_result = self._evaluate(configuration, cache, dump2disk)

            evaluation_result.pop('fitted_pipeline')

            _logger.info(f"END Evaluation of template {self.template.template['name']} {hash(str(configuration))} in {current_process()}")
        except Exception as exc:
            raise RuntimeError(f'Failed template {self.template.template["name"]}') from exc

        return evaluation_result
        # assert hasattr(evaluation_result['fitted_pipeline'], 'runtime'), \
        #     'Eval does not have runtime'

        # try:
        #     evaluation_result = self._evaluate(configuration, cache, dump2disk)
        # except:
        #     traceback.print_exc()
        #     return None
        # configuration.data.update(new_data)

    def _evaluate(self,
                  configuration: ConfigurationPoint,
                  cache: PrimitivesCache,
                  dump2disk: bool = True) -> typing.Dict:

        start_time = time.time()
        pipeline = self.template.to_pipeline(configuration)
        # Todo: update ResourceManager to run pipeline:  ResourceManager.add_pipeline(pipeline)
        # initlize repeat_time_level
        self._repeat_times_level_2 = 1
        self._repeat_times_level_1 = 1

        # for timeseries forcasting, we can't compare directly
        if self.problem['problem']['task_type'] == TaskType.TIME_SERIES_FORECASTING:
            # just skip for now
            # TODO: add one way to evalute time series forecasting pipeline quality
            # (something like sliding window)
            fitted_pipeline = FittedPipeline(
                pipeline=pipeline,
                dataset_id=self.train_dataset1.metadata.query(())['id'],
                metric_descriptions=self.performance_metrics,
                template=self.template, problem=self.problem, extra_primitive=self.extra_primitive, random_seed=self.random_seed)
            fitted_pipeline.fit(cache=cache, inputs=[self.train_dataset1])
            fitted_pipeline.save(self.output_directory)

            training_ground_truth = get_target_columns(self.train_dataset1)

            # fake_metric = calculate_score(training_ground_truth, training_ground_truth,
            #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)

            fake_metric = score_prediction(training_ground_truth, [self.train_dataset1], self.problem, self.performance_metrics, self.random_seed)

            # HACK, if mean_base_line then make it slightly worse
            if fitted_pipeline.template_name == 'SRI_Mean_Baseline_Template':
                result = fake_metric[0]
                if result['metric'].best_value() < result['metric'].worst_value():
                    result['value'] = result['value'] + 0.1
                    fake_metric[0].normalize(result['value'])
                else:
                    result['value'] = result['value'] - 0.1
                    fake_metric[0].normalize(result['value'])

            fitted_pipeline.set_metric(fake_metric[0])

            # [{'column_name': 'Class', 'metric': 'f1', 'value': 0.1}]
            data = {
                # 2019-7-10: return pipeline.id as id to make debugging easier
                'id': fitted_pipeline.pipeline.id,
                'fid': fitted_pipeline.id,
                'fitted_pipeline': fitted_pipeline,
                'training_metrics': fake_metric,
                'cross_validation_metrics': None,
                'test_metrics': fake_metric,
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
                'ensemble_tuning_result': None,
                'ensemble_tuning_metrics': None,
            }

            fitted_pipeline.auxiliary = dict(data)
            fitted_pipeline.save(self.output_directory)
            return data

        # following codes should only for running in the normal validation that can be splitted and tested
        # if in cross validation mode
        if self.testing_mode == Mode.CROSS_VALIDATION_MODE:
            self._repeat_times_level_2 = int(self.validation_config['cross_validation'])
            # start training and testing
            fitted_pipeline = FittedPipeline(
                pipeline=pipeline,
                dataset_id=self.train_dataset1.metadata.query(())['id'],
                metric_descriptions=self.performance_metrics,
                template=self.template, problem=self.problem, extra_primitive=self.extra_primitive, random_seed=self.random_seed)

            fitted_pipeline.fit(cache=cache, inputs=[self.train_dataset1])

            training_prediction = fitted_pipeline.get_fit_step_output(self.template.get_output_step_number())
            # training_ground_truth = get_target_columns(self.train_dataset1)
            # training_metrics = calculate_score(training_ground_truth, training_prediction,
            #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
            training_metrics = score_prediction(training_prediction, [self.train_dataset1], self.problem, self.performance_metrics, self.random_seed)

            cv_metrics = fitted_pipeline.get_cross_validation_metrics()
            test_metrics = copy.deepcopy(training_metrics)

            # use cross validation's avg value as the test score
            for i in range(len(test_metrics)):
                test_metrics[i]["value"] = cv_metrics[i]["value"]

            _logger.info("CV finish")

        # if in normal testing mode(including default testing mode with train/test one time each)
        else:
            # update: 2019.3.19
            # no need to run inside(level 2 split), run base on level 1 split now!
            if self.testing_mode == Mode.TRAIN_TEST_MODE:
                self._repeat_times_level_1 = int(self.validation_config['test_validation'])

            _logger.info("Will use normal train-test mode ( n ={}) to choose best primitives.".format(self._repeat_times_level_2))

            training_metrics = []
            test_metrics = []

            for each_repeat in range(self._repeat_times_level_2):
                # start training and testing
                fitted_pipeline = FittedPipeline(
                    pipeline=pipeline,
                    dataset_id=self.train_dataset2[each_repeat].metadata.query(())['id'],
                    metric_descriptions=self.performance_metrics,
                    template=self.template, problem=self.problem, extra_primitive=self.extra_primitive, random_seed=self.random_seed)

                fitted_pipeline.fit(cache=cache, inputs=[self.train_dataset2[each_repeat]])
                # fitted_pipeline.fit(inputs=[self.train_dataset2[each_repeat]])
                training_prediction = fitted_pipeline.get_fit_step_output(
                    self.template.get_output_step_number())

                # training_ground_truth = get_target_columns(self.train_dataset2[each_repeat])
                # training_metrics_each = calculate_score(
                #     training_ground_truth, training_prediction,
                #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
                training_metrics_each = score_prediction(training_prediction, [self.train_dataset2[each_repeat]],
                                                         self.problem, self.performance_metrics, self.random_seed)

                # only do test if the test_dataset exist
                if self.test_dataset2[each_repeat] is not None:
                    results = fitted_pipeline.produce(inputs=[self.test_dataset2[each_repeat]])
                    # Note: results == test_prediction
                    test_prediction = fitted_pipeline.get_produce_step_output(
                        self.template.get_output_step_number())

                    # test_ground_truth = get_target_columns(self.test_dataset2[each_repeat])
                    # test_metrics_each = calculate_score(test_ground_truth, test_prediction,
                    #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
                    test_metrics_each = score_prediction(test_prediction, [self.test_dataset2[each_repeat]],
                                                         self.problem, self.performance_metrics, self.random_seed)

                else:
                    # test_ground_truth = None
                    test_prediction = None
                    test_metrics_each = copy.deepcopy(training_metrics_each)
                    for each in test_metrics_each:
                        each["value"] = each['metric'].worst_value()

                training_metrics.append(training_metrics_each)
                test_metrics.append(test_metrics_each)
            # END for TRAIN_TEST_MODES
            # sample format of the output
            # [{'metric': 'f1Macro', 'value': 0.48418535913661614, 'values': [0.4841025641025641,
            #  0.4841025641025641, 0.4843509492047203]]
            # modify the test_metrics and training_metrics format to fit the requirements
            # print("[INFO] Testing finish.!!!")

            if len(training_metrics) > 1:
                training_metrics = self.conclude_k_fold_metrics(training_metrics)
            else:
                if type(training_metrics[0]) is list:
                    training_metrics = training_metrics[0]

            if len(test_metrics) > 1:
                test_metrics = self.conclude_k_fold_metrics(test_metrics)
            else:
                if type(test_metrics[0]) is list:
                    test_metrics = test_metrics[0]
        # END evaluation part

        # Save results
        ensemble_tuning_result = None
        ensemble_tuning_metrics = None
        if self.test_dataset1 is None:
            # print("The dataset no need to split of split failed, will not train again.")
            fitted_pipeline2 = fitted_pipeline
            # set the metric for calculating the rank
            fitted_pipeline2.set_metric(training_metrics[0])
            cv = fitted_pipeline2.get_cross_validation_metrics()
            if not cv:
                # CandidateCache asserts cv must be a list
                cv = []

            data = {
                # 2019-7-10: return pipeline.id as id to make debugging easier
                'id': fitted_pipeline2.pipeline.id,
                'fid': fitted_pipeline2.id,
                'fitted_pipeline': fitted_pipeline2,
                'training_metrics': training_metrics,
                'cross_validation_metrics': cv,
                'test_metrics': training_metrics,
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
                'ensemble_tuning_result': ensemble_tuning_result,
                'ensemble_tuning_metrics': ensemble_tuning_metrics,
            }
            fitted_pipeline.auxiliary = dict(data)

            # print("!!!! No test_dataset1")
            # pprint(data)
            # print("!!!!")

            if _logger.getEffectiveLevel() <= 10:
                data_to_logger_info = []
                if 'metric' in data['test_metrics']:
                    data_to_logger_info.append(data['test_metrics']['metric'])
                else:
                    data_to_logger_info.append("No test metrics metric found")
                if 'value' in data['test_metrics']:
                    data_to_logger_info.append(data['test_metrics']['value'])
                else:
                    data_to_logger_info.append("No test metrics value found")
                _logger.info(
                    'fitted id: %(fitted_pipeline_id)s, metric: %(metric)s, value: %(value)s',
                    {
                        'fitted_pipeline_id': fitted_pipeline2.id,
                        'metric': data_to_logger_info[0],
                        'value': data_to_logger_info[1]
                    })

            # Save fitted pipeline
            pickled = False
            if self.output_directory is not None and dump2disk:
                try:
                    fitted_pipeline2.save(self.output_directory)
                    pickled = True
                except Exception as e:
                    _logger.warning(f'SKIPPING Pickle test. Saving pipeline failed: {e.message}')

            # Pickle test
            try:
                if pickled and self.output_directory is not None and dump2disk:
                    _logger.debug("Test pickled pipeline. id: {}".format(fitted_pipeline2.id))
                    self.test_pickled_pipeline(
                        folder_loc=self.output_directory,
                        pipeline_id=fitted_pipeline2.id,
                        test_dataset=self.train_dataset2[0],
                        test_metrics=training_metrics
                        # test_ground_truth=get_target_columns(self.train_dataset2[0], self.problem)
                    )
            except Exception as e:
                _logger.exception('Pickle test Failed', exc_info=True)
        else:
            # update v2019.3.17, running k-fold corss validation on level_1 split
            if self.quick_mode:
                _logger.info("[INFO] Now in quick mode, will skip training with train_dataset1")
                # if in quick mode, we did not fit the model with dataset_train1 again
                # just generate the predictions on dataset_test1 directly and get the rank
                fitted_pipeline2 = fitted_pipeline
                fitted_pipeline2.produce(inputs=[self.test_dataset1])
                test_prediction = fitted_pipeline2.get_produce_step_output(
                    self.template.get_output_step_number())

                # test_ground_truth = get_target_columns(self.test_dataset1)
                # test_metrics2 = calculate_score(test_ground_truth, test_prediction,
                #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
                test_metrics2 = score_prediction(test_prediction, [self.test_dataset1],
                                                 self.problem, self.performance_metrics, self.random_seed)

            else:
                _logger.info("[INFO] Now in normal mode, will add extra train with train_dataset1")
                # otherwise train again with dataset_train1 and get the rank

                if self._repeat_times_level_1 > 1:
                    # generate split base on level 1 (do all-dataset level x-fold corss vaidation)
                    from common_primitives.kfold_split import KFoldDatasetSplitPrimitive, Hyperparams as hyper_k_fold
                    hyperparams_split = hyper_k_fold.defaults()
                    hyperparams_split = hyperparams_split.replace({"number_of_folds":self._repeat_times_level_1, "shuffle":True})
                    if self.task_type == 'CLASSIFICATION':
                        hyperparams_split = hyperparams_split.replace({"stratified":True})
                    else:# if not task_type == "REGRESSION":
                        hyperparams_split = hyperparams_split.replace({"stratified":False})
                    split_primitive = KFoldDatasetSplitPrimitive(hyperparams = hyperparams_split)
                    split_primitive.set_training_data(dataset = self.all_dataset)
                    split_primitive.fit()
                    query_dataset_list = list(range(self._repeat_times_level_1))
                    train_return = split_primitive.produce(inputs = query_dataset_list).value#['learningData']
                    test_return = split_primitive.produce_score_data(inputs = query_dataset_list).value

                    all_test_metrics = []
                    for i in range(self._repeat_times_level_1):
                        current_train_dataset = train_return[i]
                        current_test_dataset = test_return[i]
                        fitted_pipeline2 = FittedPipeline(
                            pipeline=pipeline,
                            dataset_id=current_train_dataset.metadata.query(())['id'],
                            metric_descriptions=self.performance_metrics,
                            template=self.template, problem=self.problem, extra_primitive=self.extra_primitive, random_seed=self.random_seed)
                        # retrain and compute ranking/metric using self.train_dataset
                        # fitted_pipeline2.fit(inputs = [self.train_dataset1])
                        fitted_pipeline2.fit(cache=cache, inputs=[current_train_dataset])
                        fitted_pipeline2.produce(inputs=[current_test_dataset])
                        test_prediction = fitted_pipeline2.get_produce_step_output(self.template.get_output_step_number())

                        # test_ground_truth = get_target_columns(current_test_dataset)
                        # test_metrics_temp = calculate_score(test_ground_truth, test_prediction,
                        #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
                        test_metrics_temp = score_prediction(test_prediction, [current_test_dataset],
                                                             self.problem, self.performance_metrics, self.random_seed)

                        all_test_metrics.append(test_metrics_temp)

                    results = self.conclude_k_fold_metrics(all_test_metrics)
                    test_metrics2 = results[0]
                else:
                    # otherwise still do as previously
                    fitted_pipeline2 = FittedPipeline(
                        pipeline=pipeline,
                        dataset_id=self.train_dataset1.metadata.query(())['id'],
                        metric_descriptions=self.performance_metrics,
                        template=self.template, problem=self.problem, extra_primitive=self.extra_primitive, random_seed=self.random_seed)
                    # retrain and compute ranking/metric using self.train_dataset
                    # fitted_pipeline2.fit(inputs = [self.train_dataset1])
                    fitted_pipeline2.fit(cache=cache, inputs=[self.train_dataset1])
                    fitted_pipeline2.produce(inputs=[self.test_dataset1])
                    test_prediction = fitted_pipeline2.get_produce_step_output(self.template.get_output_step_number())

                    # test_ground_truth = get_target_columns(self.test_dataset1)
                    # test_metrics2 = calculate_score(test_ground_truth, test_prediction,
                    #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
                    test_metrics2 = score_prediction(test_prediction, [self.test_dataset1],
                                                     self.problem, self.performance_metrics, self.random_seed)
            # update here:
            # Now new version of d3m runtime don't allow to run ".fit()" again on a given runtime
            #  object second time
            # So here we need to create a new FittedPipeline object to run second time's
            # runtime.fit()

            fitted_pipeline_final = FittedPipeline(
                pipeline=pipeline,
                dataset_id=self.all_dataset.metadata.query(())['id'],
                metric_descriptions=self.performance_metrics,
                template=self.template, problem=self.problem, extra_primitive = self.extra_primitive, random_seed=self.random_seed)
            # set the metric for calculating the rank
            fitted_pipeline_final.set_metric(test_metrics2[0])
            # end uptdate v2019.3.17

            # finally, fit the model with all data and save it
            _logger.info(
                "[INFO] Now are training the pipeline with all dataset and saving the pipeline.")
            fitted_pipeline_final.fit(cache=cache, inputs=[self.all_dataset])

            if self.ensemble_tuning_dataset:
                fitted_pipeline_final.produce(inputs=[self.ensemble_tuning_dataset])
                ensemble_tuning_result = fitted_pipeline_final.get_produce_step_output(self.template.get_output_step_number())

                # ensemble_tuning_result_ground_truth = get_target_columns(self.ensemble_tuning_dataset)
                # ensemble_tuning_metrics = calculate_score(ensemble_tuning_result_ground_truth, ensemble_tuning_result,
                #                                           self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
                ensemble_tuning_metrics = score_prediction(ensemble_tuning_result, [self.ensemble_tuning_dataset],
                                                           self.problem, self.performance_metrics, self.random_seed)

            cv = fitted_pipeline_final.get_cross_validation_metrics()
            if not cv:
                # CandidateCache asserts cv must be a list
                cv = []
            data = {
                # 2019-7-10: return pipeline.id as id to make debugging easier
                'id': fitted_pipeline_final.pipeline.id,
                'fid': fitted_pipeline_final.id,
                'fitted_pipeline': fitted_pipeline_final,
                'training_metrics': training_metrics,
                'cross_validation_metrics': cv,
                'test_metrics': test_metrics2,
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
                'ensemble_tuning_result': ensemble_tuning_result,
                'ensemble_tuning_metrics': ensemble_tuning_metrics,
            }
            fitted_pipeline.auxiliary = dict(data)

            # Save fiteed pipeline
            pickled = False
            if self.output_directory is not None and dump2disk:
                try:
                    fitted_pipeline_final.save(self.output_directory)
                    pickled = True
                except Exception as e:
                    _logger.warning('SKIPPING Pickle test. Saving pipeline failed: {e.message}')

            # Pickle test
            if pickled and self.output_directory is not None and dump2disk:
                try:
                    # remove the augmented columns in self.test_dataset1 to ensure we can pass the picking test
                    res_id, test_dataset1_df = d3m_utils.get_tabular_resource(dataset=self.test_dataset1, resource_id=None)

                    original_columns = []
                    remained_columns_number = 0
                    for i in range(test_dataset1_df.shape[1]):
                        current_selector = (res_id, ALL_ELEMENTS, i)
                        meta = self.test_dataset1.metadata.query(current_selector)

                        if AUGMENTED_COLUMN_SEMANTIC_TYPE in meta['semantic_types'] or Q_NODE_SEMANTIC_TYPE in meta['semantic_types']:
                            self.test_dataset1.metadata = self.test_dataset1.metadata.remove(selector=current_selector)
                        else:
                            original_columns.append(i)
                            if remained_columns_number != i:
                                self.test_dataset1.metadata = self.test_dataset1.metadata.remove(selector=current_selector)
                                updated_selector = (res_id, ALL_ELEMENTS, remained_columns_number)
                                self.test_dataset1.metadata = self.test_dataset1.metadata.update(selector=updated_selector, metadata=meta)
                            remained_columns_number += 1

                    self.test_dataset1[res_id] = self.test_dataset1[res_id].iloc[:, original_columns]
                    meta = dict(self.test_dataset1.metadata.query((res_id, ALL_ELEMENTS)))
                    dimension = dict(meta['dimension'])
                    dimension['length'] = remained_columns_number
                    meta['dimension'] = frozendict.FrozenOrderedDict(dimension)
                    self.test_dataset1.metadata = self.test_dataset1.metadata.update((res_id, ALL_ELEMENTS), frozendict.FrozenOrderedDict(meta))
                    # end removing augmente columns

                    _ = fitted_pipeline_final.produce(inputs=[self.test_dataset1])
                    test_prediction3 = fitted_pipeline_final.get_produce_step_output(
                        self.template.get_output_step_number())

                    # test_ground_truth_for_test_pickle = get_target_columns(self.test_dataset1)
                    # test_metrics3 = calculate_score(test_ground_truth_for_test_pickle, test_prediction3,
                    #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
                    test_metrics3 = score_prediction(test_prediction3, [self.test_dataset1],
                                                     self.problem, self.performance_metrics, self.random_seed)

                    _logger.info("Test pickled pipeline. id: {}".format(fitted_pipeline_final.id))
                    self.test_pickled_pipeline(folder_loc=self.output_directory,
                                               pipeline_id=fitted_pipeline_final.id,
                                               test_dataset=self.test_dataset1,
                                               test_metrics=test_metrics3
                                               # test_ground_truth=test_ground_truth_for_test_pickle
                    )
                except Exception as e:
                    _logger.exception('Pickle test Failed', exc_info=True)

        # still return the original fitted_pipeline with relation to train_dataset1
        return data

    def conclude_k_fold_metrics(self, input_metrics:typing.List):
        metric_value_dict: typing.Dict[list] = collections.defaultdict(list)
        # convert for test matrics
        for each in input_metrics:
            # for condition only one exist?
            if type(each) is dict:
                metric_value_dict[each['column_name']].append(each['value'])

            else:
                if len(each) > 1:
                    _logger.error("???Check here please!!!!")
                for each_target in each:
                    metric_value_dict[each_target['column_name']].append(each_target['value'])

        # test_metrics part
        output_metrics_new = input_metrics[0]
        count = 0
        for (k, v) in metric_value_dict.items():
            output_metrics_new[count]['value'] = sum(v) / len(v)
            output_metrics_new[count]['values'] = v
            count += 1
        output_metrics = [output_metrics_new]
        return output_metrics

    def test_pickled_pipeline(self, folder_loc: str, pipeline_id: str, test_dataset: Dataset,
                              test_metrics: typing.List) -> None:

        fitted_pipeline = FittedPipeline.load(folder_loc=folder_loc, fitted_pipeline_id=pipeline_id)
        results = fitted_pipeline.produce(inputs=[test_dataset])

        pipeline_prediction = fitted_pipeline.get_produce_step_output(
            self.template.get_output_step_number())
        pipeline_prediction = graph_problem_conversion(self.task_type, pipeline_prediction)

        # test_pipeline_metrics2 = calculate_score(test_ground_truth, pipeline_prediction,
        #     self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
        test_pipeline_metrics2 = score_prediction(pipeline_prediction, [test_dataset], self.problem, self.performance_metrics, self.random_seed)

        _logger.info(f'=== original:{test_metrics}')
        # _logger.info(f'=== test:{test_pipeline_metrics}')
        _logger.info(f'=== test2:{test_pipeline_metrics2}')

        pairs = zip(test_metrics, test_pipeline_metrics2)
        if any(x != y for x, y in pairs):
            warn("[WARN] Test pickled pipeline mismatch. id: {}".format(fitted_pipeline.id))
            print(
                {
                    'id': fitted_pipeline.id,
                    'test__metric': test_metrics,
                    'pickled_pipeline__metric': test_pipeline_metrics2
                }
            )
            print("\n" * 5)
            _logger.warning(
                "Test pickled pipeline mismatch. 'id': '%(id)s', 'test__metric': '%("
                "test__metric)s', 'pickled_pipeline__metric': '%(pickled_pipeline__metric)s'.",
                {
                    'id': fitted_pipeline.id,
                    'test__metric': test_metrics,
                    'pickled_pipeline__metric': test_pipeline_metrics2
                },
            )
            print(
                "Test pickled pipeline mismatch. 'id': '%(id)s', 'test__metric': '%("
                "test__metric)s', 'pickled_pipeline__metric': '%("
                "pickled_pipeline__metric)s'.".format(
                    {
                        'id': fitted_pipeline.id,
                        'test__metric': test_metrics,
                        'pickled_pipeline__metric': test_pipeline_metrics2
                    })
            )
            print("\n" * 5)
        else:
            _logger.debug(("\n" * 5) + "Pickling succeeded" + ("\n" * 5))
