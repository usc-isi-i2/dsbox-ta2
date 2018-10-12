import traceback
import logging
import time
import typing
import random
from multiprocessing import Pool
from pprint import pprint

import pandas as pd
from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.combinatorial_search.TemplateSpaceParallelBaseSearch import \
    TemplateSpaceParallelBaseSearch
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


class MultiBanditSearch(TemplateSpaceParallelBaseSearch[T]):
    max_trials_per_template: int = 3
    """
    Use multi armed bandit to select the templates then search through the template randomly for
    a while
    """

    def __init__(self, template_list: typing.List[DSBoxTemplate],
                 performance_metrics: typing.List[typing.Dict],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 output_directory: str, log_dir: str, timeout: int = 55, num_proc: int = 4) -> None:

        # Use first metric from test
        TemplateSpaceParallelBaseSearch.__init__(
            self=self,
            template_list=template_list,
            performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1,
            train_dataset2=train_dataset2, test_dataset1=test_dataset1,
            test_dataset2=test_dataset2, all_dataset=all_dataset,
            log_dir=log_dir, output_directory=output_directory, timeout=timeout, num_proc=num_proc)

        # UCT scores holder
        # self.uct_score = dict(map(lambda t: (t, None), template_list))

    def _select_next_template(self, num_iter=2) -> \
            typing.Tuple[ConfigurationSpaceBaseSearch, str]:

        # initial evaluation
        for search in self.confSpaceBaseSearch:
            # yield search, "random"
            yield search, "random"

        # UCT based evaluation
        for _ in range(num_iter):
            # while True:
            _choices, _weights = self._update_UCT_score()
            selected = random_choices_without_replacement(_choices, _weights, 1)
            yield selected[0], "random"

    def _update_UCT_score(self) -> typing.Tuple[ConfigurationSpaceBaseSearch, typing.Dict]:

        try:
            normalize = self.history.normalize()
        except TypeError:
            _logger.exception(traceback.format_exc())
            print(self.history)
            exit(1)
        uct_score = {}
        for t_name in normalize:
            uct_score[t_name] = BanditDimensionalSearch.compute_UCT(normalize[t_name])

        self._logger.info(f"UCT updated: {uct_score}")

        valid_templates = [k for k, v in uct_score.items() if v is not None]

        _choices = self.confSpaceBaseSearch[valid_templates]
        _weights = uct_score[valid_templates]

        return _choices, _weights

    def search(self, num_iter: int=2) -> typing.Dict:
        """
        runs the random search for each compatible template and returns the report of the best
        template evaluated. In each iteration the method randomly sample one of the templates
        from the template list based on their UCT score and runs random search on the template.
        Args:
            num_iter:
            Number of iterations of dim search.
        Returns:
            the report related to the best template (only the evaluated templates not the whole
            list)
        """
        # the actual search goes here
        self._search_templates(num_iter=num_iter)

        # cleanup the caches and cache manager
        self.cacheManager.cleanup()

        # cleanup job manager
        self.job_manager.kill_job_mananger()

        return self.history.get_best_history()

    def _search_templates(self, num_iter: int = 2) -> None:
        """
        runs the random search for each compatible template and returns the report of the best
        template evaluated. In each iteration the method randomly sample one of the templates
        from the template list based on their UCT score and runs random search on the template.
        Args:
            num_iter:

        Returns:

        """
        template_iter = self._select_next_template(num_iter=num_iter)
        for i, (search, mode) in enumerate(template_iter):
            print("#" * 50)
            print(f"[INFO] Selected Template: {search.template.template['name']}")
            print("$" * 100)
            self._random_pipeline_evaluation_push(search=search,
                                                  num_iter=self.job_manager.proc_num)
            if (i+2)%3 == 0:
                self._get_evaluation_results(max_num=self.job_manager.proc_num)
                print("$" * 100)
                print(self.history)
                print("$" * 100)

        self._get_evaluation_results()