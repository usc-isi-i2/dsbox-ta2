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

from dsbox.combinatorial_search.search_utils import random_choices_without_replacement
from dsbox.combinatorial_search.search_utils import get_target_columns
from dsbox.combinatorial_search.search_utils import Status
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.template.pipeline_utilities import pipe2str

import importlib
colorama_spec = importlib.util.find_spec("colorama")
STYLE = ""
ERROR = ""
WARNING = ""
if colorama_spec is not None:
    from colorama import Fore, Back, init

    # STYLE = Fore.BLUE + Back.GREEN
    STYLE = Fore.BLACK + Back.GREEN
    ERROR = Fore.WHITE + Back.RED
    WARNING = Fore.BLACK + Back.YELLOW
    init(autoreset=True)


T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class TemplateDimensionalSearch(ConfigurationSpaceBaseSearch[PrimitiveDescription]):
    """
    Use dimensional search to find best pipeline.

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
                 template: typing.List[DSBoxTemplate],
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
        super().__init__(template_list=template, configuration_space=configuration_space,
                         train_dataset1=train_dataset1, train_dataset2=train_dataset2,
                         test_dataset1=test_dataset1, test_dataset2=test_dataset2,
                         all_dataset=all_dataset, problem=problem,
                         performance_metrics=performance_metrics, log_dir=log_dir,
                         output_directory=output_directory)

        self.num_workers = os.cpu_count() if num_workers == 0 else num_workers

        # if not set(self.template.template_nodes.keys()) <= set(configuration_space.get_dimensions()):
        #     raise exceptions.InvalidArgumentValueError(
        #         "Not all template steps are in configuration space: {}".format(self.template.template_nodes.keys()))

    def _lookup_candidate(self, candidate: ConfigurationPoint[T], cand_cache: typing.Dict) -> \
            typing.Tuple[ConfigurationPoint[T], float]:

        key = hash(str(candidate))
        if key in cand_cache:
            line = cand_cache[key]
            print("[INFO] hit@Candidate: ({},{})".format(key, line["id"]))
            # print("[INFO] candidate cache Hit@{}:{}".format(key, line['candidate']))
            return line['candidate'], line['value']
        else:
            return (None, None)

    def _push_candidate(self, result: typing.Dict, candidate: ConfigurationPoint[T],
                        cand_cache: typing.Dict) -> None:
        key = hash(str(candidate))
        cand_id = result['fitted_pipeline'].id if result else None
        value = result['test_metrics'][0]['value'] if result else None
        # add value to candidate cache

        if self._is_candidate_hit(candidate, cand_cache):
            assert value == cand_cache[key]['value'], \
                "New value for candidate:" + str(candidate)
            return

        print("[INFO] push@Candidate: ({},{})".format(key, cand_id))
        cand_cache[key] = {
            "candidate": candidate,
            "id": cand_id,
            "value": value,
        }

    def _is_candidate_hit(self, candidate: ConfigurationPoint[T], cand_cache: typing.Dict) -> bool:
        return hash(str(candidate)) in cand_cache

    def search_one_iter(self, candidate_in: ConfigurationPoint[T] = None,
                        max_per_dimension: int = 50,
                        cache_bundle: typing.Tuple[typing.Dict, typing.Dict] = (None, None)) -> \
            typing.Dict:
        """
        Performs one iteration of dimensional search. During dimesional
        search our algorithm iterates through all 8 steps of pipeline as
        indicated in our configuration space and greedily optimizes the
        pipeline one step at a time.

        Parameters
        ----------
        candidate_in: ConfigurationPoint[T]
            Current best candidate
        max_per_dimension: int
            Maximum number of values to search per dimension
        cache_bundle: tuple[Dict,Dict]
            the global cache object and candidate cache object for storing and reusing computation
            on intermediate results. if the cache is None, a local cache will be used in the
            dimensional search
        """

        # setup the output cache
        local_cache = False
        if cache_bundle[0] is None or cache_bundle[1] is None:
            print("[INFO] Using Local Cache")
            local_cache = True
            manager = Manager()
            cache = manager.dict()
            candidate_cache = manager.dict()
        else:
            print("[INFO] Using Global Cache")
            cache, candidate_cache = cache_bundle

        # initialize the simulation counter
        sim_counter = 0
        start_time = time.clock()

        # we first need the baseline for searching the conf_space. For this
        # purpose we initially use first configuration and evaluate it on the
        #  dataset. In case that failed we repeat the sampling process one
        # more time to guarantee robustness on error reporting

        try:
            candidate, candidate_value = \
                self._setup_initial_candidate(candidate_in, cache, candidate_cache)
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
            selected = random_choices_without_replacement(choices, weights, max_per_dimension)

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

    def search_template(self, template: DSBoxTemplate, candidate: typing.Dict = None,
                        cache_bundle: typing.Tuple[typing.Dict, typing.Dict] = (None, None)) \
            -> typing.Dict:

        self._logger.info('Searching template %s', template.template['name'])

        space = template.generate_configuration_space()

        metrics = self.problem['problem']['performance_metrics']

        # # setup the dimensional search configs
        # search = TemplateDimensionalSearch(
        #     template=template, configuration_space=space, problem=self.problem_doc_metadata,
        #     test_dataset1=self.test_dataset1, train_dataset1=self.train_dataset1,
        #     test_dataset2=self.test_dataset2, train_dataset2=self.train_dataset2,
        #     all_dataset=self.all_dataset, performance_metrics=metrics,
        #     output_directory=self.output_directory, log_dir=self.output_logs_dir,
        #     num_workers=self.num_cpus
        # )

        self.minimize = search.minimize
        # candidate, value = search.search_one_iter()
        self._logger.info('cache size = {}'.format(len(cache_bundle[0])))
        report = search.search_one_iter(candidate_in=candidate, cache_bundle=cache_bundle)
        candidate = report['candidate']
        value = report['best_val']
        # assert "fitted_pipe" in candidate, "argument error!"
        if candidate is None:
            self._logger.error("[ERROR] not candidate!")
            return report  # return Status.PROBLEM_NOT_IMPLEMENT
        else:
            self._logger.info("******************\n[INFO] Writing results")
            pprint.pprint(candidate.data)
            self._logger.info(str(candidate.data) + " " + str(value))
            if candidate.data['training_metrics']:
                self._logger.info('Training {} = {}'.format(
                    candidate.data['training_metrics'][0]['metric'],
                    candidate.data['training_metrics'][0]['value']))
            if candidate.data['cross_validation_metrics']:
                self._logger.info('CV {} = {}'.format(
                    candidate.data['cross_validation_metrics'][0]['metric'],
                    candidate.data['cross_validation_metrics'][0]['value']))
            if candidate.data['test_metrics']:
                self._logger.info('Validation {} = {}'.format(
                    candidate.data['test_metrics'][0]['metric'],
                    candidate.data['test_metrics'][0]['value']))

            # FIXME: code used for doing experiments, want to make optionals
            # pipeline = FittedPipeline.create(configuration=candidate,
            #                             dataset=self.dataset)

            # dataset_name = self.output_executables_dir.rsplit("/", 2)[1]
            # # save_location = os.path.join(self.output_logs_dir, dataset_name + ".txt")
            # save_location = self.output_directory + ".txt"

            # self._logger.info("******************\n[INFO] Saving training results in %s",
            # save_location)
            # try:
            #     f = open(save_location, "w+")
            #     f.write(str(metrics) + "\n")

            #     for m in ["training_metrics", "cross_validation_metrics", "test_metrics"]:
            #         if m in candidate.data and candidate.data[m]:
            #             f.write(m + ' ' +  str(candidate.data[m][0]['value']) + "\n")
            #     # f.write(str(candidate.data['training_metrics'][0]['value']) + "\n")
            #     # f.write(str(candidate.data['cross_validation_metrics'][0]['value']) + "\n")
            #     # f.write(str(candidate.data['test_metrics'][0]['value']) + "\n")
            #     f.close()
            # except:
            #     self._logger.exception('[ERROR] Save training results Failed!')
            #     raise NotSupportedError(
            #         '[ERROR] Save training results Failed!')

            return report

    def run(self):
        """
        runs the dim search for each compatible template and returns the best trained pipeline
        for the problem.
        Returns:
            fittedPipeline: the best fittedpipeline
        """
        if not self.template:
            return Status.PROBLEM_NOT_IMPLEMENT

        # setup the output cache
        manager = Manager()
        cache = manager.dict()
        candidate_cache = manager.dict()
        for i in range(10):
            pass




PythonPathWithHyperaram = typing.Tuple[PythonPath, int, HyperparamDirective]