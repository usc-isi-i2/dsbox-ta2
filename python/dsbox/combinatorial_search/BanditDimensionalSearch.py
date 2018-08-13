import os
import random
import traceback
import typing
import logging
import time
import copy
import sys

from warnings import warn

from multiprocessing import Pool, current_process, Manager
from pprint import pprint

from d3m.exceptions import NotSupportedError
from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from d3m.metadata.base import ALL_ELEMENTS
from d3m.metadata.problem import PerformanceMetric

from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.pipeline.utils import larger_is_better
from dsbox.schema.problem import optimization_type
from dsbox.schema.problem import OptimizationType

from dsbox.template.template import DSBoxTemplate
from dsbox.template.template import HyperparamDirective

from dsbox.template.configuration_space import DimensionName
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.configuration_space import ConfigurationSpace

from dsbox.combinatorial_search.search_utils import random_choices_without_replacement, \
    get_target_columns
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.template.pipeline_utilities import pipe2str

from dsbox.combinatorial_search.RandomDimensionalSearch import RandomDimensionalSearch


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

    def __init__(self,
                 template: DSBoxTemplate,
                 configuration_space: ConfigurationSpace[PrimitiveDescription],
                 problem: Metadata,
                 train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset],
                 test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset],
                 all_dataset: Dataset,
                 performance_metrics: typing.List[typing.Dict],
                 output_directory: str,
                 log_dir: str,
                 num_workers: int = 0) -> None:

        # Use first metric from test
        super().__init__(template=template, configuration_space=configuration_space,
                         train_dataset1=train_dataset1, train_dataset2=train_dataset2,
                         test_dataset1=test_dataset1, test_dataset2=test_dataset2,
                         all_dataset=all_dataset, problem=problem,
                         performance_metrics=performance_metrics, log_dir=log_dir,
                         output_directory=output_directory, num_workers=num_workers)

    def select_next_template(self, max_iter: int = 2):
        # while True:
        choices = list(range(len(self.template)))

        # initial evaluation
        for i in choices:
            yield i

        # print("[INFO] Choices:", choices)
        # UCT based evaluation
        for i in range(max_iter):
            # while True:
            valids = list(filter(lambda t: t[1] is not None,
                                 zip(choices, self.uct_score)))
            _choices = list(map(lambda t: t[0], valids))
            _weights = list(map(lambda t: t[1], valids))
            selected = random_choices_without_replacement(_choices, _weights, 1)
            yield selected[0]

    def update_UCT_score(self, index: int, report: typing.Dict):
        self.update_history(index, report)

        alpha = 0.01
        self.normalize = self.exec_history[['reward', 'exe_time', 'trial']]
        scale = (self.normalize.max() - self.normalize.min())
        scale.replace(to_replace=0, value=1, inplace=True)
        self.normalize = (self.normalize - self.normalize.min()) / scale
        self.normalize.clip(lower=0.01, upper=1, inplace=True)

        for i in range(len(self.uct_score)):
            self.uct_score[i] = self.compute_UCT(i)

        self._logger.info(STYLE + "[INFO] UCT updated: %s", self.uct_score)

    def update_history(self, index, report):
        self.total_run += report['sim_count']
        self.total_time += report['time']
        row = self.exec_history.iloc[index]
        update = {
            'trial': row['trial'] + report['sim_count'],
            'exe_time': row['exe_time'] + report['time'],
            'candidate': report['candidate'],
        }
        if report['reward'] is not None:
            update['reward'] = (
                    (row['reward'] * row['trial'] + report['reward'] * report['sim_count']) /
                    (row['trial'] + report['sim_count'])
            )
            update['best_value'] = max(report['reward'], row['best_value'])

        for k in update:
            self.exec_history.iloc[index][k] = update[k]

    def compute_UCT(self, index=0):
        beta = 10
        gamma = 1
        delta = 4
        history = self.normalize.iloc[index]
        try:

            reward = history['reward']
            # / history['trial']

            return (beta * (reward) * max(log(10 * history['trial']), 1) +
                    gamma * sqrt(2 * log(self.total_run) / history['trial']) +
                    delta * sqrt(2 * log(self.total_time) / history['exe_time']))
        except:
            self._logger.error('Failed to compute UCT. Defaulting to None')
            # print(STYLE+"[WARN] compute UCT failed:", history.tolist())
            return None

    def initialize_uct(self):
        self.total_run = 0
        self.total_time = 0
        # self.exec_history = \
        #     [{"exe_time": 1, "reward": 1, "trial": 1, "candidate": None, "best_value": 0}] * \
        #     len(self.template)

        self.exec_history = pd.DataFrame(None,
                                         index=map(lambda s: s.template["name"], self.template),
                                         columns=['reward', 'exe_time', 'trial', 'candidate',
                                                  'best_value'])
        self.exec_history[['reward', 'exe_time', 'trial']] = 0
        self.exec_history[['best_value']] = float('-inf')

        self.exec_history['candidate'] = self.exec_history['candidate'].astype(object)
        self.exec_history['candidate'] = None

        # print(self.exec_history.to_string())
        self.uct_score = [None] * len(self.template)





PythonPathWithHyperaram = typing.Tuple[PythonPath, int, HyperparamDirective]