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


class RandomDimensionalSearch(TemplateSpaceParallelBaseSearch[T]):
    max_init_trials: int = 3
    """
    Use dimensional search across randomly chosen templates to find best pipeline.
    """

    def __init__(self, template_list: typing.List[DSBoxTemplate],
                 performance_metrics: typing.List[typing.Dict],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 ensemble_tuning_dataset:Dataset,
                 output_directory: str, log_dir: str, timeout: int = 55, num_proc: int = 4) -> None:

        # Use first metric from test
        TemplateSpaceParallelBaseSearch.__init__(
            self=self,
            template_list=template_list,
            performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1,
            train_dataset2=train_dataset2, test_dataset1=test_dataset1,
            test_dataset2=test_dataset2, all_dataset=all_dataset, ensemble_tuning_dataset = ensemble_tuning_dataset,
            log_dir=log_dir, output_directory=output_directory, timeout=timeout, num_proc=num_proc)

    def search_one_iter(self, search: ConfigurationSpaceBaseSearch, max_per_dim: int = 50,) -> None:
        """
        Performs one iteration of dimensional search. During dimesional search our algorithm
        iterates through all steps of pipeline as indicated in our configuration space and
        greedily optimizes the pipeline one step at a time.

        Args:
        search: ConfigurationSpaceBaseSearch
            the template to run dim search on
        max_per_dim: int
            Maximum number of values to search per dimension

        Returns:
            None
        """

        # we first need the baseline for searching the conf_space. For this
        # purpose we initially use first configuration and evaluate it on the
        #  dataset. In case that failed we repeat the sampling process one
        # more time to guarantee robustness on error reporting
        try:
            candidate_report = \
                self._setup_initial_candidate(base_search=search)
            base_candidate = candidate_report['configuration']
        except ValueError:
            return None

        # generate an executable pipeline with random steps from conf. space.
        # The actual searching process starts here.
        for dimension in search.configuration_space.get_dimension_search_ordering():
            new_candidates = self._generate_dimensional_search_list(
                dimension=dimension, base_candidate=base_candidate, search=search,
                max_per_dim=max_per_dim
            )
            if len(new_candidates) <= 1:
                continue
            print("[INFO] Running Pool for step", dimension, ", fork_num:", len(new_candidates))
            _logger.info(
                "Running Pool for step {} fork_num: {}".format(dimension, len(new_candidates)))
            print("-"*50)
            # send all the candidates to the execution Engine
            for conf in new_candidates:

                # push the candidate to the job manager
                # push the candidate to the job manager
                self.job_manager.push_job(
                    kwargs_bundle=self._prepare_job_posting(candidate=conf,
                                                            search=search)
                )
                time.sleep(0.1)

            # wait until all the candidates are evaluated
            self._get_evaluation_results()

            base_candidate = self.history.get_best_candidate(search.template.template['name'])

        # END FOR

    def _generate_dimensional_search_list(self, dimension: str,
                                          search: ConfigurationSpaceBaseSearch,
                                          base_candidate: ConfigurationPoint,
                                          max_per_dim: int) -> typing.List[ConfigurationPoint]:
        """
        Samples the configuration space in dimensional fashion. In this type of sampling the
        primitives in all steps except the 'dimension' are similar to the base_candidate. The
        primitive in the 'dimension' is selected randomly for each one of the sampled candidates.
        The method makes sure that all the sampled candidates have not been evaluated before to
        prevent duplicate evaluation. The assumption is if the duplicate evaluations are not
        better than the base candidate hence there is no need to even check their result.
        Args:
            dimension: str
                the dimension that we perform dim_search on
            search: ConfigurationSpaceBaseSearch
                the ConfigurationSpaceBaseSearch we do dim_search in
            base_candidate: ConfigurationPoint
                the base candidate (best candidate so far) that is being used as basis of dim search
            max_per_dim: int
                maximum samples per dimension

        Returns:
            List of all the unique sampled candidates that need to be evaluated
        """
        # get all possible choices for the step, as specified in
        # configuration space
        choices: typing.List[T] = search.configuration_space.get_values(dimension)

        if len(choices) == 1:  # if only have one candidate primitive for this step, skip?
            return []
        # print("[INFO] choices:", choices, ", in step:", dimension)
        assert 1 < len(choices), \
            f'Step {dimension} has no primitive choices!'

        # the weights are assigned by template designer
        weights = [search.configuration_space.get_weight(dimension, x) for x in choices]

        # generate the candidates choices list
        selected = random_choices_without_replacement(choices, weights, max_per_dim)

        # No need to evaluate if value is already known
        if base_candidate[dimension] in selected:
            selected.remove(base_candidate[dimension])

        # all the new possible pipelines are generated in new_candidates
        new_candidates: typing.List[ConfigurationPoint] = []
        for value in selected:
            # transfer the pipeline to dictionary type so that we can change detail steps
            new = dict(base_candidate)
            # replace the traget step
            new[dimension] = value
            # regenerate the pipeline
            candidate_ = search.configuration_space.get_point(new)

            if self._prepare_candidate_4_eval(candidate=candidate_):
                new_candidates.append(candidate_)
            # if not self.cacheManager.candidate_cache.is_hit(candidate_):
            #     new_candidates.append(candidate_)

        return new_candidates

    def _setup_initial_candidate(self, base_search: ConfigurationSpaceBaseSearch) -> typing.Tuple:
        """
        we first need the baseline for searching the conf_space. For this purpose we initially
        use first configuration and evaluate it on the dataset. In case that failed we repeat the
        sampling process one more time to guarantee robustness on error reporting

        Args:
            base_search: confSpaceBaseSearch
                the base configuration space search object. It contains the template, conf space
                and the evaluate method neccessary for evaluating the confpoints from the template

        Returns:
            report: typing.Dict
        """

        _logger.info("setting up initial candidate")

        template = base_search.template
        candidate = self.history.get_best_candidate(template.template['name'])

        if candidate is None:
            candidate = base_search.configuration_space.get_random_assignment()

        # first, then random, then another random
        for i in range(RandomDimensionalSearch.max_init_trials):
            try:
                report = self.evaluate_blocking(base_search=base_search, candidate=candidate)
                if report is None:
                    raise ValueError("Initial Pipeline failed, Trying a random pipeline")
                return report
            except:
                traceback.print_exc()
                _logger.error(traceback.format_exc())
                _logger.warning('Initial Pipeline failed, Trying a random pipeline ...')
                # print("[WARN] Initial Pipeline failed, Trying a random pipeline ...")
                pprint(candidate)
                print("-" * 20)
                candidate = base_search.configuration_space.get_random_assignment()

        raise ValueError("No valid initial candidates found")


    def search(self, num_iter: int=2) -> typing.Dict:
        """
        runs the dim search for each compatible template and returns the report of the best
        template evaluated. In each iteration the method randomly sample one of the templates
        from the template list and runs dim-search on the template.
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
        for search in self._select_next_template(num_iter=num_iter):
            print("#" * 50)
            print(f"[INFO] Selected Template: {search.template.template['name']}")
            print("$" * 100)
            print("$" * 100)
            self.search_one_iter(search=search, max_per_dim=5)
            print("$" * 100)
            print(self.history)
            print("$" * 100)
