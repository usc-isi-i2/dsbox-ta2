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


class BanditDimensionalSearch(RandomDimensionalSearch):
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
                 output_directory: str, log_dir: str, timeout: int = 55, num_proc: int = 4) -> None:

        # Use first metric from test
        RandomDimensionalSearch.__init__(
            self=self,
            template_list=template_list,
            performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1,
            train_dataset2=train_dataset2, test_dataset1=test_dataset1,
            test_dataset2=test_dataset2, all_dataset=all_dataset,
            log_dir=log_dir, output_directory=output_directory, timeout=timeout, num_proc=num_proc)

        # UCT scores holder
        self.uct_score = dict(map(lambda t: (t, None), template_list))


    def _select_next_template(self, num_iter=2) -> \
            typing.Tuple[ConfigurationSpaceBaseSearch, str]:

        # initial evaluation
        for search in self.confSpaceBaseSearch:
            # yield search, "random"
            yield search, "dimensional"

        # UCT based evaluation
        for i in range(num_iter):
            # while True:
            _choices, _weights = self._update_UCT_score()
            selected = random_choices_without_replacement(_choices, _weights, 1)
            yield selected[0], "dimensional"

    def _update_UCT_score(self) -> typing.Tuple[ConfigurationSpaceBaseSearch, typing.Dict]:

        normalize = self.history.normalize()

        for t_name in normalize:
            self.uct_score[t_name] = BanditDimensionalSearch.compute_UCT(normalize[t_name])

        self._logger.info(STYLE + "[INFO] UCT updated: %s", self.uct_score)

        valid_templates = list(filter(lambda t: self.uct_score[t] is not None, self.uct_score))

        _choices = self.confSpaceBaseSearch[valid_templates]
        _weights = self.uct_score[valid_templates]

        return _choices, _weights

    def _search_templates(self, num_iter: int = 2) -> None:
        for search, mode in self._select_next_template(num_iter=num_iter):
            print("#" * 50)
            print(f"[INFO] Selected Template: {search.template.template['name']}")
            print("$" * 100)
            print("$" * 100)

            if mode == "random":
                self._template_random_investigation(search=search,
                                                    num_iter=self.job_manager.proc_num)
            if mode == "dimensional":
                self.search_one_iter(search=search, max_per_dim=self.job_manager.proc_num)

            print("$" * 100)
            print(self.history)
            print("$" * 100)

    def _template_random_investigation(self, search: ConfigurationSpaceBaseSearch,
                                       num_iter: int=1) -> None:

        self._random_pipeline_sampling(search=search, num_iter=num_iter)

        self._get_evaluation_results(template_name=search.template.template['name'])

    @staticmethod
    def compute_UCT(history: typing.Union[pd.Series, pd.DataFrame, typing.Dict]):
        beta = 10
        gamma = 1
        delta = 4
        try:
            # / history['trial']

            return (beta * (history['reward']) * max(log(10 * history['trial']), 1) +
                    gamma * history['trial']) / sqrt(2 * log(self.total_run) +
                    delta * sqrt(2 * log(self.total_time) / history['exe_time']))
        except (KeyError, ZeroDivisionError):
            self._logger.error('Failed to compute UCT. Defaulting to None')
            # print(STYLE+"[WARN] compute UCT failed:", history.tolist())
            return None

