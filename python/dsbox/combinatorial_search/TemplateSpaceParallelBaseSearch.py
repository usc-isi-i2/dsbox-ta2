import importlib
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
from dsbox.combinatorial_search.cache import PrimitivesCache
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.template import DSBoxTemplate

# from dsbox.template.pipeline_utilities import pipe2str
spam_spec = importlib.util.find_spec("colorama")
STYLE = ""
ERROR = ""
WARNING = ""
if spam_spec is not None:
    from colorama import Fore, Back, init

    # STYLE = Fore.BLUE + Back.GREEN
    STYLE = Fore.BLACK + Back.GREEN
    ERROR = Fore.WHITE + Back.RED
    WARNING = Fore.BLACK + Back.YELLOW

    if 'PYCHARM_HOSTED' in os.environ:
        convert = False  # in PyCharm, we should disable convert
        strip = False
    else:
        convert = None
        strip = None

    init(autoreset=True, convert=convert, strip=strip)

T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class TemplateSpaceParallelBaseSearch(TemplateSpaceBaseSearch[T]):
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

    def __init__(self, template_list: typing.List[DSBoxTemplate],
                 performance_metrics: typing.List[typing.Dict],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 output_directory: str, log_dir: str, timeout: int=55, num_proc: int=4) -> None:

        TemplateSpaceBaseSearch.__init__(self=self,
            template_list=template_list, performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1, train_dataset2=train_dataset2,
            test_dataset1=test_dataset1, test_dataset2=test_dataset2, all_dataset=all_dataset,
            output_directory=output_directory, log_dir=log_dir
        )

        # DistributedJobManager.__init__(proc_num=num_proc, timeout=timeout)
        self.job_manager = DistributedJobManager(proc_num=num_proc, timeout=timeout)

    @staticmethod
    def _evaluate_template(confspace_search: ConfigurationSpaceBaseSearch,
                           candidate: ConfigurationPoint, cache: PrimitivesCache,
                           dump2disk: bool = True):
        return confspace_search.evaluate_pipeline(args=(candidate, cache, dump2disk))

    def search(self, num_iter=1):
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
        self.job_manager.start_workers(target=self._evaluate_template)
        time.sleep(0.1)

        # randomly send the candidates to job manager for evaluation
        self._push_random_candidates(num_iter)

        time.sleep(0.1)

        # iteratively wait until a result is available and process the result untill there is no
        # other pending job in the job manager
        self._get_evaluation_results()

        # cleanup the caches and cache manager
        self.cacheManager.cleanup()

        self.job_manager.kill_job_mananger()
        return self.bestResult

    def _get_evaluation_results(self):
        print(STYLE + "[INFO] Waiting for the results")
        while not self.job_manager.is_idle():
            (kwargs, report) = self.job_manager.pop_job(block=True)
            candidate = kwargs['candidate']
            try:
                if report is None:
                    raise ValueError("Search Failed on candidate")
                self._update_best_result(report)
                self.cacheManager.candidate_cache.push(report)
            except:
                traceback.print_exc()
                print(ERROR + "[INFO] Search Failed on candidate")
                pprint(candidate)
                self.cacheManager.candidate_cache.push_None(candidate=candidate)

    def _push_random_candidates(self, num_iter):
        for i in range(num_iter):
            print("#" * 50)
            template_index = random.randrange(0, len(self.confSpaceBaseSearch))
            search = self.confSpaceBaseSearch[template_index]
            candidate = search.configuration_space.get_random_assignment()
            print(STYLE + "[INFO] Selecting Template:", search.template.template['name'])

            if self.cacheManager.candidate_cache.is_hit(candidate):
                report = self.cacheManager.candidate_cache.lookup(candidate)
                assert report is not None and 'configuration' in report, \
                    'invalid candidate_cache line: {}->{}'.format(candidate, report)
            else:
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