import logging
import random
import time
import traceback
import typing

from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.combinatorial_search.TemplateSpaceBaseSearch import TemplateSpaceBaseSearch
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.combinatorial_search.ExecutionHistory import ExecutionHistory
from dsbox.JobManager.cache import CacheManager
from dsbox.template.template import DSBoxTemplate

from dragonfly import minimise_function, maximise_function
from dragonfly.opt.gp_bandit import bo_from_func_caller
from dragonfly.exd.worker_manager import RealWorkerManager
from dragonfly.utils.reporters import get_reporter

from typing import Any as Any_t, Dict as Dict_t, List as List_t, Tuple
from typing import Tuple as Tuple_t

T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)

Dragonfly_config_t = typing.Dict[str, typing.Union[str, float, int]]

class DragonflySearch(TemplateSpaceBaseSearch):
    """
    Search the template space through the individual configuration spaces
    using the dragonfly optimizer. Dragonfly uses Bayesing Optimization to
    handle primitive selection and hyperparamenter tuning simultaneously.

    Attributes
    ----------
    drgnfly_config_list : List[Dragonfly_config_t]
        list of configurations in dragonfly required format. Each config file is
        associated to a template.

    TODO:
        1. Update confspace class to handle translation from dragonfly config
        file
        2. updating history
        3. test the config translation code
    """

    MAX_CAPITAL = 1*60*60

    def initialize_problem(self, template_list: typing.List[DSBoxTemplate],
                           performance_metrics: typing.List[typing.Dict],
                           problem: Metadata, train_dataset1: Dataset,
                           train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                           test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                           ensemble_tuning_dataset: Dataset,
                           output_directory: str, log_dir: str,
                           start_time: float = 0,
                           timeout_sec: float = -1) -> None:

        super().initialize_problem(
            template_list=template_list,
            performance_metrics=performance_metrics,
            problem=problem, train_dataset1=train_dataset1,
            train_dataset2=train_dataset2,
            test_dataset1=test_dataset1, test_dataset2=test_dataset2,
            all_dataset=all_dataset,
            ensemble_tuning_dataset=ensemble_tuning_dataset,
            output_directory=output_directory, log_dir=log_dir,
            start_time=start_time, timeout_sec=timeout_sec
        )

        self.drgnfly_config_list: typing.List[Dragonfly_config_t] = list(
            map(lambda t:
                DSBoxTemplate_Dragonfly.get_drgnfly_config(t), template_list))


    def search(self, num_iter=1, *, one_pipeline_only=False) -> typing.Dict[str, typing.Any]:
        """
        This method implements the random search method with support of multiple templates. The
        method incorporates the primitives cache and candidates cache to store the intermediate
        results.
        Args:
            num_iter:
                number of iterations of random sampling
        Returns:

        """
        EXP_DIR = 'experiment_dir_%s' % (time.strftime('%Y%m%d%H%M%S'))

        success_count = 0
        for drgn_conf, search in self._select_next_template(num_iter=num_iter):
            if self._done(success_count, one_pipeline_only=one_pipeline_only):
                break
            _logger.info(f'Search template {search.template}')

            # worker_manager = RealWorkerManager(GPU_IDS, EXP_DIR)

            opt_pt, opt_val, history = minimise_function(objective_to_min,
                                                         drgn_conf['domain'],
                                                         self.MAX_CAPITAL,
                                                         capital_type='realtime',
                                                         config=drgn_conf)
            # for candidate in self._sample_random_pipeline(search=search, num_iter=1):
            #     if self._done(success_count):
            #         break
            #     try:
            #         report = search.evaluate_pipeline(
            #             args=(candidate, self.cacheManager.primitive_cache, True))
            #         success_count += 1
            #         _logger.info(f'Search template pipeline SUCCEEDED {search.template}')
            #         _logger.info(f'report fitted_pipeline {report["id"]}')
            #         _logger.debug(f'report {report}')
            #     except Exception:
            #         traceback.print_exc()
            #         _logger.error(f'Search template pipeline FAILED {search.template}')
            #         _logger.error(traceback.format_exc())
            #         _logger.debug("Failed candidate: {candidate}")
            #         report = None
            #
            #     kwargs_bundle = self._prepare_job_posting(candidate=candidate,
            #                                               search=search)
            #     self._add_report_to_history(kwargs_bundle=kwargs_bundle,
            #                                 report=report)

        self.cacheManager.cleanup()
        return self.history.get_best_history()

    def _add_report_to_history(self, kwargs_bundle: typing.Dict[str, typing.Any],
                               report: typing.Dict[str, typing.Any]) -> None:
        """
        extract information from input of jobmanager (kwargs_bundle) and pipeline's evaluation
        output (report) to update evaluation history object
        Args:
            kwargs_bundle: {'target_object':..., 'target_method':...., 'kwargs':{'args':(...)}}
            report: generated report from pipeline evaluation method

        Returns:

        """
        candidate = kwargs_bundle['kwargs']['args'][0]
        template_name = kwargs_bundle['target_obj'].template.template['name']
        if report is not None:
            report['template_name'] = template_name
            _logger.info(f"New report: id={report['id']}")
            _logger.debug(f"Report details: {report}")
            self.history.update(report, template_name=template_name)
            self.cacheManager.candidate_cache.push(report)
        else:
            _logger.info(f"Search Failed on candidate {hash(str(candidate))}")
            _logger.warning(traceback.format_exc())
            self.history.update_none(fail_report=None, template_name=template_name)
            self.cacheManager.candidate_cache.push_None(candidate=candidate)

    def _select_next_template(self, num_iter: int = 2) \
            -> Tuple_t[Dragonfly_config_t, ConfigurationSpaceBaseSearch]:
        """
        Selects a confSpace (template) randomly from the list of available ones
        Args:
            num_iter: number of samples to draw

        Returns:
            generator containing confSpace objects

        """
        for _ in range(num_iter):
            drgn_config, search = random.choice(zip(self.drgnfly_config_list,
                                       self.confSpaceBaseSearch))
            yield drgn_config, search