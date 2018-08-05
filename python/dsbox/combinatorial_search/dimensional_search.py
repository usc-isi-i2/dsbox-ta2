import traceback
import logging
import time
import traceback
import typing
from multiprocessing import Pool
from pprint import pprint

import pandas as pd
from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.combinatorial_search.TemplateSpaceParallelBaseSearch import \
    TemplateSpaceParallelBaseSearch
from dsbox.combinatorial_search.search_utils import random_choices_without_replacement
from dsbox.combinatorial_search.search_utils import ExecutionHistory
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.template import DSBoxTemplate
from dsbox.template.template import HyperparamDirective

T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class TemplateRandomDimensionalSearch(TemplateSpaceParallelBaseSearch[T]):
    """
    Use dimensional search to find best pipeline.

    Attributes:
        template:
        configuration_space:
        problem:
        train_dataset1:
        train_dataset2:
        test_dataset1:
        test_dataset2:
        all_dataset:
        performance_metrics:
        output_directory:
        log_dir:
        num_workers:
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

        self.exec_history: ExecutionHistory = ExecutionHistory(
            template_list=template_list, is_better=TemplateRandomDimensionalSearch._is_better)


    def search_one_iter(self, candidate_in: ConfigurationPoint[T],
                        confspace_search: ConfigurationSpaceBaseSearch,
                        max_per_dim: int = 50,) -> typing.Dict:
        """
        Performs one iteration of dimensional search. During dimesional search our algorithm
        iterates through all steps of pipeline as indicated in our configuration space and
        greedily optimizes the pipeline one step at a time.

        Parameters
        ----------
        candidate_in: ConfigurationPoint[T]
            Current best candidate
        max_per_dim: int
            Maximum number of values to search per dimension
        """

        # initialize the simulation counter
        sim_counter = 0
        start_time = time.clock()

        # we first need the baseline for searching the conf_space. For this
        # purpose we initially use first configuration and evaluate it on the
        #  dataset. In case that failed we repeat the sampling process one
        # more time to guarantee robustness on error reporting

        try:
            candidate, candidate_value = \
                self._setup_initial_candidate(candidate_in)
        except:
            report_failed = {
                'reward': None,
                'time': time.clock() - start_time,
                'sim_count': 3,
                'candidate': None,
                'best_val': float('-inf')
            }
            # return (candidate, candidate_value)
            return report_failed

        sim_counter += 1
        # generate an executable pipeline with random steps from conf. space.
        # The actual searching process starts here.
        for dimension in self.dimension_ordering:
            # get all possible choices for the step, as specified in
            # configuration space
            choices: typing.List[T] = self.configuration_space.get_values(dimension)

            # TODO this is just a hack

            if len(choices) == 1:  # if only have one candidate primitive for this step, skip?
                continue
            # print("[INFO] choices:", choices, ", in step:", dimension)
            assert 1 < len(choices), \
                f'Step {dimension} has not primitive choices!'

            # the weights are assigned by template designer
            weights = [self.configuration_space.get_weight(dimension, x) for x in choices]

            # generate the candidates choices list
            selected = random_choices_without_replacement(choices, weights, max_per_dim)

            score_values = []
            sucessful_candidates = []

            # No need to evaluate if value is already known
            if candidate_value is not None and candidate[dimension] in selected:
                selected.remove(candidate[dimension])
                # also add this candidate to the succesfully_candidates to make comparisons easier
                sucessful_candidates.append(candidate)
                score_values.append(candidate_value)

            # all the new possible pipelines are generated in new_candidates
            new_candidates: typing.List[ConfigurationPoint] = []
            for value in selected:
                # transfer the pipeline to dictionary type so that we can change detail steps
                new = dict(candidate)
                # replace the traget step
                new[dimension] = value
                # regenerate the pipeline
                candidate_ = self.configuration_space.get_point(new)

                if not self._is_candidate_hit(candidate_, candidate_cache):
                    new_candidates.append(candidate_)

            best_index = -1
            print('*' * 100)
            print("[INFO] Running Pool for step", dimension, ", fork_num:", len(new_candidates))
            sim_counter += len(new_candidates)
            # run all candidate pipelines in multi-processing mode
            try:
                with Pool(self.num_workers) as p:
                    results = p.map(
                        self.evaluate_pipeline,
                        map(lambda c: (c, cache), new_candidates)
                    )
                for res, x in zip(results, new_candidates):
                    self._push_candidate(res, x, candidate_cache)
                    if not res:
                        print('[ERROR] candidate failed:')
                        pprint(x)
                        print("-" * 10)
                        continue

                    score_values.append(res['test_metrics'][0]['value'])
                    cross_validation_mode = False

                    # pipeline = self.template.to_pipeline(x)
                    # res['pipeline'] = pipeline

                    res['fitted_pipeline'] = res['fitted_pipeline']
                    x.data.update(res)
                    sucessful_candidates.append(x)
            except:
                traceback.print_exc()

            # If all candidates failed, only the initial one in the score_values
            if len(score_values) == 1:
                print("[INFO] No new Candidate worked in this step!")
                if not candidate:
                    print("[ERROR] The template did not return any valid pipelines!")
                    return (None, None)
                else:
                    continue
            # Find best candidate
            if self.minimize:
                best_index = score_values.index(min(score_values))
            else:
                best_index = score_values.index(max(score_values))

            # # for conditions that no test dataset given or in the new training mode
            # if sum(test_values) == 0 and cross_validation_values:
            #     best_index = best_cv_index
            if cross_validation_mode:
                print("[INFO] Best index:", best_index, " --> CV matrix score:",
                      score_values[best_index])
            else:
                print("[INFO] Best index:", best_index, " --> Test matrix score:",
                      score_values[best_index])

            # put the best candidate pipeline and results to candidate
            candidate = sucessful_candidates[best_index]
            candidate_value = score_values[best_index]
        # END FOR

        # shutdown the cache manager
        if local_cache:
            manager.shutdown()

        # here we can get the details of pipelines from "candidate.data"
        assert "fitted_pipeline" in candidate.data, "parameters not added! last"
        # gather information for UCT

        reward = candidate_value

        # TODO : come up with a better method for this
        if self.minimize:
            reward = -reward

        report = {
            'reward': reward,
            'time': time.clock() - start_time,
            'sim_count': sim_counter,
            'candidate': candidate,
            'best_val': candidate_value
        }
        # return (candidate, candidate_value)
        return report

    def _setup_initial_candidate(self,
                                 candidate: ConfigurationPoint[T],
                                 cache: typing.Dict,
                                 candidate_cache: typing.Dict) -> \
            typing.Tuple[ConfigurationPoint[T], float]:
        """
        we first need the baseline for searching the conf_space. For this
        purpose we initially use first configuration and evaluate it on the
        dataset. In case that failed we repeat the sampling process one more
        time to guarantee robustness on error reporting

        Args:
            candidate: ConfigurationPoint[T]
                the previous best candidate. In case it is None, random candidate will be
                evaluated. if the random candidates fail on the dataset the process of choosing
                a random candidate will be repeated three times

            cache: typing.Dict
                cache object for storing intermediate primitive outputs

            candidate_cache: typing.Dict
                cache object for storing candidate evalueation results

        Returns:
            candidate, evaluate_value : ConfigurationPoint[T], float
        """
        if candidate is None:
            candidate = self.configuration_space.get_first_assignment()
            #ConfigurationPoint(self.configuration_space, self.first_assignment())
        # first, then random, then another random
        for i in range(2):
            try:
                cand_tmp, value_tmp = self._lookup_candidate(candidate, candidate_cache)

                # if the candidate has been evaluated before and its metric value was None (
                # meaning it was not compatible with dataset), then reevaluating the candidate is
                #  redundant
                # if cand_tmp is not None a compatible with dataset), then reevaluating the
                # candidate is redundant
                if cand_tmp is not None and value_tmp is None:
                    raise ValueError("Candidate is not compatible with the dataset")

                result = self.evaluate_pipeline((candidate, cache, True if cand_tmp is None else False))

                self._push_candidate(result, candidate, candidate_cache)

                candidate.data.update(result)

                if 'cross_validation_metrics' in result and len(
                        result['cross_validation_metrics']) > 0:
                    return (candidate, result['cross_validation_metrics'][0]['value'])
                else:
                    return (candidate, result['test_metrics'][0]['value'])
            except:
                traceback.print_exc()
                print("[ERROR] Initial Pipeline failed, Trying a random pipeline ...")
                pprint(candidate)
                print("-" * 20)
                candidate = self.configuration_space.get_random_assignment()
                #ConfigurationPoint(self.configuration_space, self.random_assignment())
        raise ValueError("Invalid initial candidate")

    def search(self, num_iter: int=2):
        pass
    # def search(self):
    #     """
    #     runs the dim search for each compatible template and returns the best trained pipeline
    #     for the problem.
    #     Returns:
    #         fittedPipeline: the best fittedpipeline
    #     """
    #     if not self.template:
    #         return Status.PROBLEM_NOT_IMPLEMENT
    #
    #     # setup the output cache
    #     manager = Manager()
    #     cache = manager.dict()
    #     candidate_cache = manager.dict()
    #     for i in range(10):
    #         pass




PythonPathWithHyperaram = typing.Tuple[PythonPath, int, HyperparamDirective]