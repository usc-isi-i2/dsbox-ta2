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
                 output_directory: str, log_dir: str, timeout: int=55, num_proc: int=4) -> None:

        self.job_manager = DistributedJobManager(proc_num=num_proc, timeout=timeout)

        TemplateSpaceBaseSearch.__init__(
            self=self,
            template_list=template_list, performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1, train_dataset2=train_dataset2,
            test_dataset1=test_dataset1, test_dataset2=test_dataset2, all_dataset=all_dataset,
            output_directory=output_directory, log_dir=log_dir
        )


    @staticmethod
    def _evaluate_template(confspace_search: ConfigurationSpaceBaseSearch,
                           candidate: ConfigurationPoint, cache: PrimitivesCache,
                           dump2disk: bool = True):
        return confspace_search.evaluate_pipeline(args=(candidate, cache, dump2disk))

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
        self.setup_exec_history()
        # start the worker processes
        self.job_manager.start_workers(target=self._evaluate_template)
        time.sleep(0.1)

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

    def _get_evaluation_results(self):
        print("[INFO] Waiting for the results")
        counter = 0
        # FIXME this is_idle method is not working properly
        while not self.job_manager.is_idle():
            print("[INFO] Sleeping,", counter)
            (kwargs, report) = self.job_manager.pop_job(block=True)
            candidate = kwargs['candidate']
            try:
                if report is None:
                    raise ValueError("Search Failed on candidate")
                self.history.update(report)
                self.cacheManager.candidate_cache.push(report)
            except:
                traceback.print_exc()
                print("[INFO] Search Failed on candidate")
                pprint(candidate)
                self.history.update_none(fail_report=None)
                self.cacheManager.candidate_cache.push_None(candidate=candidate)
            counter += 1
        print("[INFO] No more pending job")

    def _push_random_candidates(self, num_iter):
        for i in range(num_iter):
            print("#" * 50)
            template_index = random.randrange(0, len(self.confSpaceBaseSearch))
            search = self.confSpaceBaseSearch[template_index]
            candidate = search.configuration_space.get_random_assignment()
            print("[INFO] Selecting Template:", search.template.template['name'])

            if self.cacheManager.candidate_cache.is_hit(candidate):
                report = self.cacheManager.candidate_cache.lookup(candidate)
                assert report is not None and 'candidate' in report, \
                    'invalid candidate_cache line: {}->{}'.format(candidate, report)
                continue

            try:
                # first we just add the candidate as failure to the candidates cache to
                # prevent it from being evaluated again while it is being evaluated
                self.cacheManager.candidate_cache.push_None(candidate=candidate)

                # push the candidate to the job manager
                self.job_manager.push_job(
                    {
                        'confspace_search': search,
                        'cache': self.cacheManager.primitive_cache,
                        'candidate': candidate,
                        'dump2disk': True,
                    })
            except:
                traceback.print_exc()

            time.sleep(0.1)

