import importlib
import logging
import operator
import os
import random
import traceback
import typing
from pprint import pprint

from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.combinatorial_search.ExecutionHistory import ExecutionHistory
from dsbox.JobManager.cache import CacheManager
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

    def __init__(self, template_list: typing.List[DSBoxTemplate],
                 performance_metrics: typing.List[typing.Dict],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 output_directory: str, log_dir: str,) -> None:

        self.template_list = template_list

        self.configuration_space_list = list(
            map(lambda t: t.generate_configuration_space(), template_list))

        self.confSpaceBaseSearch = list(
            map(
                lambda tup: ConfigurationSpaceBaseSearch(
                    template=tup[0],
                    configuration_space=tup[1],
                    problem=problem, train_dataset1=train_dataset1, train_dataset2=train_dataset2,
                    test_dataset1=test_dataset1, test_dataset2=test_dataset2,
                    all_dataset=all_dataset, performance_metrics=performance_metrics,
                    output_directory=output_directory, log_dir=log_dir
                ),
                zip(template_list, self.configuration_space_list)
            )
        )

        self.cacheManager = CacheManager()

        self.history = ExecutionHistory()

        # load libraries with a dummy evaluation
        self.confSpaceBaseSearch[0].dummy_evaluate()

    def search(self, num_iter=1) -> typing.Dict[str, typing.Any]:
        """
        This method implements the random search method with support of multiple templates. The
        method incorporates the primitives cache and candidates cache to store the intermediate
        results.
        Args:
            num_iter:
                number of iterations of random sampling
        Returns:

        """
        for i in range(num_iter):
            print("#"*50)
            search = random.choice(self.confSpaceBaseSearch)
            candidate = search.configuration_space.get_random_assignment()
            print("[INFO] Selecting Template:", search.template.template['name'])

            if self.cacheManager.candidate_cache.is_hit(candidate):
                report = self.cacheManager.candidate_cache.lookup(candidate)
                assert report is not None and 'configuration' in report, \
                    'invalid candidate_cache line: {}->{}'.format(candidate, report)
            else:
                try:
                    report = search.evaluate_pipeline(
                        args=(candidate, self.cacheManager.primitive_cache, True))
                    self.history.update(report)
                    self.cacheManager.candidate_cache.push(report)
                except:
                    traceback.print_exc()
                    print("[INFO] Search Failed on candidate")
                    pprint(candidate)
                    self.history.update_none(fail_report=None)
                    self.cacheManager.candidate_cache.push_None(candidate=candidate)

        self.cacheManager.cleanup()
        return self.history.get_best_history()
