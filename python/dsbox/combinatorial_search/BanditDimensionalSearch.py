import traceback
import logging
import time
import traceback
import typing
import random
from multiprocessing import Pool
from pprint import pprint
from math import sqrt, log

import pandas as pd
from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.combinatorial_search.TemplateSpaceParallelBaseSearch import \
    TemplateSpaceParallelBaseSearch
from dsbox.combinatorial_search.RandomDimensionalSearch import RandomDimensionalSearch
from dsbox.combinatorial_search.MultiBanditSearch import MultiBanditSearch
from dsbox.combinatorial_search.search_utils import random_choices_without_replacement
from dsbox.combinatorial_search.ExecutionHistory import ExecutionHistory
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.template import DSBoxTemplate
from dsbox.template.template import HyperparamDirective

T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class BanditDimensionalSearch(RandomDimensionalSearch, MultiBanditSearch):
    """
    Use dimensional search with UCT to find best pipeline.

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

    def __init__(self, template_list: typing.List[DSBoxTemplate],
                 performance_metrics: typing.List[typing.Dict],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 ensemble_tuning_dataset:Dataset,
                 output_directory: str, log_dir: str, timeout: int = 55, num_proc: int = 4) -> None:

        # Use first metric from test
        RandomDimensionalSearch.__init__(
            self=self,
            template_list=template_list,
            performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1,
            train_dataset2=train_dataset2, test_dataset1=test_dataset1,
            test_dataset2=test_dataset2, all_dataset=all_dataset,
            ensemble_tuning_dataset = ensemble_tuning_dataset,
            log_dir=log_dir, output_directory=output_directory, timeout=timeout, num_proc=num_proc)

        MultiBanditSearch.__init__(
            self=self,
            template_list=template_list,
            performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1,
            train_dataset2=train_dataset2, test_dataset1=test_dataset1,
            test_dataset2=test_dataset2, all_dataset=all_dataset,
            ensemble_tuning_dataset = ensemble_tuning_dataset,
            log_dir=log_dir, output_directory=output_directory, timeout=timeout, num_proc=num_proc)

    # def search(self, num_iter: int = 2) -> typing.Dict:
    #     """
    #     runs the dim search for each compatible template and returns the report of the best
    #     template evaluated. In each iteration the method randomly sample one of the templates
    #     from the template list and runs dim-search on the template.
    #     Args:
    #         num_iter:
    #         Number of iterations of dim search.
    #     Returns:
    #         the report related to the best template (only the evaluated templates not the whole
    #         list)
    #     """
    #     # the actual search goes here
    #     self._search_templates(num_iter=num_iter)
    #
    #     # cleanup the caches and cache manager
    #     self.cacheManager.cleanup()
    #
    #     # cleanup job manager
    #     self.job_manager.reset()
    #
    #     return self.history.get_best_history()

    def _search_templates(self, num_iter: int = 2) -> None:
        """
        runs the random search for each compatible template and returns the report of the best
        template evaluated. In each iteration the method randomly sample one of the templates
        from the template list based on their UCT score and runs random search on the template.
        Args:
            num_iter:

        Returns:

        """
        # print("I am here _search_templates")
        max_round_per_dim = 3
        template_iter = (self._select_next_template(num_iter=num_iter))
        for i, (search, mode) in enumerate(template_iter):
            print("#" * 50)
            print(f"[INFO] Selected Template: {search.template.template['name']}")
            print("$" * 100)
            (self.search_one_iter(search=search,
                              max_per_dim=self.job_manager.proc_num * max_round_per_dim))

        self._get_evaluation_results()

    def _select_next_template(self, num_iter=2) -> \
            typing.Tuple[ConfigurationSpaceBaseSearch, str]:
        yield from self._bandit_select_next_template(num_iter)
