import logging
import os
import random
import time
import traceback
import typing

from pprint import pprint

from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from dsbox.JobManager.DistributedJobManager import DistributedJobManager
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.combinatorial_search.TemplateSpaceBaseSearch import TemplateSpaceBaseSearch
from dsbox.JobManager.cache import PrimitivesCache
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.template import DSBoxTemplate

T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class TemplateSpaceParallelBaseSearch(TemplateSpaceBaseSearch[T]):
    """
    Search the template space through random configuration spaces in parallel.

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

    def __init__(self, template_list: typing.List[DSBoxTemplate],
                 performance_metrics: typing.List[typing.Dict],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 ensemble_tuning_dataset: Dataset,
                 output_directory: str, log_dir: str, timeout: int=55, num_proc: int=4) -> None:

        self.job_manager = DistributedJobManager(proc_num=num_proc, timeout=timeout)

        TemplateSpaceBaseSearch.__init__(
            self=self,
            template_list=template_list, performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1, train_dataset2=train_dataset2,
            test_dataset1=test_dataset1, test_dataset2=test_dataset2, all_dataset=all_dataset,
            ensemble_tuning_dataset = ensemble_tuning_dataset,
            output_directory=output_directory, log_dir=log_dir
        )

        # setup the execution history to store the results of each template separately
        # self.setup_exec_history(template_list=self.template_list)

    # @staticmethod
    # def _evaluate_template(confspace_search: ConfigurationSpaceBaseSearch,
    #                        candidate: ConfigurationPoint, cache: PrimitivesCache,
    #                        dump2disk: bool = True):
    #     return confspace_search.evaluate_pipeline(args=(candidate, cache, dump2disk))

    def search(self, num_iter=1) -> typing.Dict:
        """
        This method implements the random search method with support of multiple templates using
        the parallel job manager. The method incorporates the primitives cache to store the
        intermediate results and uses the candidates cache to keep a record of evaluated pipelines.
        Args:
            num_iter:
                number of iterations of random sampling
        Returns:

        """
        # start the worker processes
        # self.job_manager._start_workers(target_method=self._evaluate_template)
        # from dsbox.template.runtime import ForkedPdb
        # ForkedPdb().set_trace()
        time.sleep(0.1)

        # def _timeout_handler(self, signum):
        #     print('Signal handler called with signal', signum)
        # with eventlet.Timeout(180, False):
            # signal.signal(signal.SIGALRM, _timeout_handler)
            # signal.alarm(3 * 60)

        # randomly send the candidates to job manager for evaluation
        self._push_random_candidates(num_iter)
        time.sleep(1)

        # iteratively wait until a result is available and process the result untill there is no
        # other pending job in the job manager
        self._get_evaluation_results()

        # cleanup the caches and cache manager
        self.cacheManager.cleanup()

        # cleanup job manager
        self.job_manager.kill_job_mananger()

        return self.history.get_best_history()


    def _get_evaluation_results(self, max_num: int=float('inf')) -> None:
        """
        The process is sleeped on jobManager's result queue until a result is ready, then it pops
        the results and updates history and candidate's cache with it. The method repeats this
        process until there are no pending jobs in the jobManager.
        Args:
            None
        Returns:
            None
        """
        _logger.debug("Waiting for the results")
        counter = 0
        while (counter < max_num) and (not self.job_manager.is_idle()):
            # print("[INFO] Sleeping,", counter)
            _logger.debug(f"Main Process Sleeping:{counter}")
            (kwargs_bundle, report) = self.job_manager.pop_job(block=True)
            _logger.warning(f"kwargs: {kwargs_bundle}")

            self._add_report_to_history(kwargs_bundle, report)

            counter += 1
        _logger.debug("[INFO] No more pending job")

    def _push_random_candidates(self, num_iter: int):
        """
        randomly samples 'num_iter' unique pipelines from a random configuration space and pushes
        them to jobManager for evaluation.
        Args:
            num_iter: number of pipeline samples

        Returns:

        """
        print("#" * 50)
        for search in self._select_next_template(num_iter=num_iter):
            self._random_pipeline_evaluation_push(search=search, num_iter=1)

        print("#" * 50)

    def _random_pipeline_evaluation_push(self, search: ConfigurationSpaceBaseSearch,
                                         num_iter: int = 1) -> None:
        """
        randomly samples 'num_iter' unique pipelines from an specified configuration space and
        pushes them to jobManager for evaluation.
        Args:
            search: the selected configuration space (template)
            num_iter: number of pipelines to sample

        Returns:

        """
        for candidate in self._sample_random_pipeline(search=search, num_iter=num_iter):

            try:
                # push the candidate to the job manager
                self.job_manager.push_job(
                    kwargs_bundle=self._prepare_job_posting(candidate=candidate,
                                                            search=search)
                )
            except:
                traceback.print_exc()
                _logger.error(traceback.format_exc())

            time.sleep(0.1)

    def evaluate_blocking(self, base_search: ConfigurationSpaceBaseSearch,
                          candidate: ConfigurationPoint[PrimitiveDescription]) -> typing.Dict:
        """
        submits the candidate to the execution engine and blocks execution until the evaluation
        is done.
        Args:
            base_search: ConfigurationSpaceBaseSearch
                the confSpaceBaseSearch that the candidate is from
            candidate: ConfigurationPoint[PrimitiveDescription]
                the candidate to be evaluated

        Returns:
            report: typing.Dict
                the evaluation result in the same format that evaluate will produce
        Warnings:
            the code assumes that no other process is reading results from the executionManger's
            output queue. If the popped job is not the same that was submitted the method will
            raise exception.

        """
        # check the cache for evaluation. If the candidate has been evaluated before and
        # its metric value was None (meaning it was not compatible with dataset),
        # then reevaluating the candidate is redundant
        if self.cacheManager.candidate_cache.is_hit(candidate):
            report = self.cacheManager.candidate_cache.lookup(candidate)
            assert report is not None and 'configuration' in report, \
                'invalid candidate_cache line: {}->{}'.format(candidate, report)

            # if cand_tmp is not None a compatible with dataset), then reevaluating the
            # candidate is redundant
            if 'value' not in report or report['value'] is None:
                raise ValueError("Initial candidate is not compatible with the dataset")

            return report

        # first we just add the candidate as failure to the candidates cache to
        # prevent it from being evaluated again while it is being evaluated
        self.cacheManager.candidate_cache.push_None(candidate=candidate)

        # push the candidate to the job manager
        # self.job_manager.push_job(
        #     {
        #         'confspace_search': base_search,
        #         'cache': self.cacheManager.primitive_cache,
        #         'candidate': candidate,
        #         'dump2disk': True,
        #     })
        self.job_manager.push_job(
            kwargs_bundle=self._prepare_job_posting(candidate=candidate,
                                                   search=base_search)
        )

        # wait for the results
        (kwargs_bundle, report) = self.job_manager.pop_job(block=True)
        check_candidate = kwargs_bundle['kwargs']['args'][0]

        if check_candidate != candidate:
            raise ValueError('Different candidate result was popped. The evaluate_blocking '
                             'assumes that it is the only process pushing jobs to jobManager')

        self._add_report_to_history(kwargs_bundle=kwargs_bundle, report=report)
        return report
