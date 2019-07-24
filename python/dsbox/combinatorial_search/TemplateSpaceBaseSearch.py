import logging
import random
import time
import traceback
import typing

import d3m.metadata.problem as problem

from d3m.container.dataset import Dataset
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.combinatorial_search.ExecutionHistory import ExecutionHistory
from dsbox.JobManager.cache import CacheManager
from dsbox.template.template import DSBoxTemplate


T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class TemplateSpaceBaseSearch(typing.Generic[T]):
    """
    Search the template space through the individual configuration spaces.

    Attributes
    ----------
    template_list : List[DSBoxTemplate]
        Evaluate given point in configuration space
    configuration_space_list: List[ConfigurationSpace[T]]
        Definition of the configuration space
    confSpaceBaseSearch: List[ConfigurationSpaceBaseSearch]
        list of ConfigurationSpaceBaseSearch related to each template
    cacheManager:  CacheManager
        the object contains the two distributed cache and their associated methods
    bestResult: typing.Dict
        the dictinary containing the results of the best pipline
    """

    def __init__(self, is_multiprocessing: bool = True):
        self.is_multiprocessing = is_multiprocessing
        self.cacheManager = CacheManager(is_multiprocessing=is_multiprocessing)

    def initialize_problem(self, template_list: typing.List[DSBoxTemplate],
                           performance_metrics: typing.List[typing.Dict],
                           problem: problem.Problem, train_dataset1: Dataset,
                           train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                           test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                           ensemble_tuning_dataset: Dataset,
                           extra_primitive: typing.Set[str],
                           output_directory: str,
                           start_time: float = 0,
                           timeout_sec: float = -1) -> None:

        # self.cacheManager.timeout_sec = timeout_sec
        self.template_list = template_list

        self.configuration_space_list = list(
            map(lambda t: t.generate_configuration_space(), template_list))

        self.confSpaceBaseSearch: typing.List[ConfigurationSpaceBaseSearch] = list(
            map(
                lambda tup: ConfigurationSpaceBaseSearch(
                    template=tup[0],
                    configuration_space=tup[1],
                    problem=problem, train_dataset1=train_dataset1, train_dataset2=train_dataset2,
                    test_dataset1=test_dataset1, test_dataset2=test_dataset2,
                    all_dataset=all_dataset, performance_metrics=performance_metrics,
                    ensemble_tuning_dataset=ensemble_tuning_dataset,
                    extra_primitive=extra_primitive,
                    output_directory=output_directory,
                ),
                zip(template_list, self.configuration_space_list)
            )
        )

        self.history: ExecutionHistory = None
        # setup the execution history to store the results of each template separately
        self._setup_exec_history(template_list=self.template_list)

        self.ensemble_tuning_result: typing.Dict = {}

        if start_time > 0:
            self.start_time = start_time
        else:
            self.start_time = time.perf_counter()
        self.timeout_sec = timeout_sec
        # load libraries with a dummy evaluation
        # try:
        #     self.confSpaceBaseSearch[-1].dummy_evaluate()
        # except:
        #     pass

    def _setup_exec_history(self, template_list: typing.List[DSBoxTemplate]):
        self.history = ExecutionHistory(template_list=template_list)

    def search(self, num_iter=1, *, one_pipeline_only=False) -> typing.Dict[str, typing.Any]:
        """
        This method implements the random search method with support of multiple templates. The
        method incorporates the primitives cache and candidates cache to store the intermediate
        results.
        Args:
            num_iter:
                number of iterations of random sampling
        Returns:

        """

        success_count = 0
        search: ConfigurationSpaceBaseSearch
        for search in self._select_next_template(num_iter=num_iter):
            if self._done(success_count, one_pipeline_only=one_pipeline_only):
                break
            _logger.info(f'Search template {search.template}')
            for candidate in self._sample_random_pipeline(search=search, num_iter=1):
                if self._done(success_count):
                    break
                try:
                    report = search.evaluate_pipeline(
                        args=(candidate, self.cacheManager.primitive_cache, True))
                    success_count += 1
                    _logger.info(f'Search template pipeline SUCCEEDED {search.template}')
                    _logger.info(f'report fitted_pipeline {report["id"]}')
                    _logger.debug(f'report {report}')
                except Exception:
                    traceback.print_exc()
                    _logger.error(f'Search template pipeline FAILED {search.template}')
                    _logger.error(traceback.format_exc())
                    _logger.debug("Failed candidate: {candidate}")
                    report = None

                kwargs_bundle = self._prepare_job_posting(candidate=candidate,
                                                          search=search)
                self._add_report_to_history(kwargs_bundle=kwargs_bundle,
                                            report=report)

        self.cacheManager.cleanup()
        return self.history.get_best_history()

    def _done(self, success_count, *, one_pipeline_only=False):
        if one_pipeline_only and success_count >= 1:
            _logger.info('Found one pipeline')
            return True
        _logger.info(f'Test Done: {time.perf_counter() > self.start_time + self.timeout_sec}: {time.perf_counter()} > {self.start_time} + {self.timeout_sec}')
        return (self.timeout_sec > 0 and time.perf_counter() > self.start_time + self.timeout_sec)

    def _add_report_to_history(self, kwargs_bundle: typing.Dict[str, typing.Any],
                               report: typing.Dict[str, typing.Any]) -> None:
        """
        extract information from input of jobmanager (kwargs_bundle) and pipeline's evaluation
        output (report) to update evaluation history object
        Args:
            kwargs_bundle: {'target_object':..., 'target_method':...., 'kwargs':{'args':(...)}}
            report: generated report from pipeline evaluation method

        Returns:

        """
        candidate = kwargs_bundle['kwargs']['args'][0]
        template_name = kwargs_bundle['target_obj'].template.template['name']
        if report is not None:
            report['template_name'] = template_name
            _logger.info(f"New report: id={report['id']}")
            _logger.debug(f"Report details: {report}")
            self.history.update(report, template_name=template_name)
            self.cacheManager.candidate_cache.push(report)
        else:
            _logger.info(f"Search Failed on candidate {hash(str(candidate))}")
            # _logger.warning(traceback.format_exc())
            self.history.update_none(fail_report=None, template_name=template_name)
            self.cacheManager.candidate_cache.push_None(candidate=candidate)

    def _prepare_job_posting(self,
                             candidate: typing.Dict[str, typing.Any],
                             search: ConfigurationSpaceBaseSearch) -> typing.Dict[str, typing.Any]:
        """
        prepares the candidate pipeline in a format that can be passed to jobManager
        Args:
            candidate: the candidate pipeline to be evaluated
            search: the confSpace (template) that pipeline is sampled from

        Returns:
            dictionary containing information for jobManager in the compatible format
        """
        return {
            'target_obj': search,
            'target_method': 'evaluate_pipeline',
            'kwargs': {
                'args': (candidate, self.cacheManager.primitive_cache, True)
            }
        }

    def _select_next_template(self, num_iter: int = 2) \
            -> typing.Iterable[ConfigurationSpaceBaseSearch]:
        """
        Selects a confSpace (template) randomly from the list of available ones
        Args:
            num_iter: number of samples to draw

        Returns:
            generator containing confSpace objects

        """
        for _ in range(num_iter):
            search = random.choice(self.confSpaceBaseSearch)
            yield search

    def _sample_random_pipeline(self,
                                search: ConfigurationSpaceBaseSearch,
                                num_iter: int = 1) \
            -> typing.Iterable[typing.Dict]:
        for _ in range(num_iter):
            candidate = search.configuration_space.get_random_assignment()

            if self._prepare_candidate_4_eval(candidate=candidate):
                _logger.info(f"Selecting Candidate: {hash(str(candidate))}")
                yield candidate
            else:
                continue

    def _prepare_candidate_4_eval(self, candidate: ConfigurationPoint) -> bool:
        """

        Args:
            candidate:

        Returns:
            Bool: whether candidate evaluation is needed or not
        """
        if self.cacheManager.candidate_cache.is_hit(candidate):
            report = self.cacheManager.candidate_cache.lookup(candidate)
            assert report is not None and 'configuration' in report, \
                'invalid candidate_cache line: {}->{}'.format(candidate, report)
            return False

        try:
            # first we just add the candidate as failure to the candidates cache to
            # prevent it from being evaluated again while it is being evaluated
            self.cacheManager.candidate_cache.push_None(candidate=candidate)
        except Exception:
            traceback.print_exc()
            _logger.error(traceback.format_exc())

        return True

    def shutdown(self):
        self.cacheManager.shutdown()
