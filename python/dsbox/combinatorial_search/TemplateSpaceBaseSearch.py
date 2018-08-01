import os
import random
import traceback
import typing
import logging
import time
import copy
import sys
import operator
from functools import reduce

from warnings import warn

from multiprocessing import Pool, current_process, Manager
from pprint import pprint

from d3m.exceptions import NotSupportedError
from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
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
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch

from dsbox.template.configuration_space import SimpleConfigurationSpace
# from dsbox.template.pipeline_utilities import pipe2str

from dsbox.combinatorial_search.cache import CacheManager
import importlib
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
        print("Hi! You are using PyCharm")
    else:
        convert = None
        strip = None

    init(autoreset=True, convert=convert, strip=strip)

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
    """

    def __init__(self, template_list: typing.List[DSBoxTemplate],
                 performance_metrics: typing.List[typing.Dict],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 output_directory: str, log_dir: str, ) -> None:

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

        self.bestResult = None


        self.cacheManager = CacheManager()


    def search(self, num_iter=1, cache=None):
        for i in range(num_iter):
            search = random.choice(self.confSpaceBaseSearch)
            candidate = search.configuration_space.get_random_assignment()
            print(STYLE+"[INFO] Selecting Template:", search.template.template['name'])
            try:
                report = search.evaluate_pipeline(args=(candidate, cache, True))
                self._update_best_result(report)
            except:
                traceback.print_exc()
                print(ERROR+"[INFO] Search Failed")
                pprint(candidate)
        return self.bestResult

    def _update_best_result(self, new_res: typing.Dict) -> None:
        if new_res is None:
            return
        pprint(new_res)
        if self.bestResult is None or \
           TemplateSpaceBaseSearch._is_better(self.bestResult, new_res):
            self.bestResult = new_res

    @staticmethod
    def _is_better(base: typing.Dict, check: typing.Dict) -> bool:
        if 'Error' in check:
            return False

        larger_is_better = ['accuracy', 'precision', 'recall', 'f1', 'f1Micro', 'f1Macro', 'rocAuc',
                            'rocAucMicro', 'rocAucMacro', 'rSquared', 'jaccardSimilarityScore',
                            'precisionAtTopK', 'objectDetectionAP', 'normalizedMutualInformation', ]
        # Larger is better
        # 'accuracy', 'precision', 'recall', 'f1', 'f1Micro', 'f1Macro', 'rocAuc',
        # 'rocAucMicro', 'rocAucMacro', 'rSquared', 'jaccardSimilarityScore',
        # 'precisionAtTopK', 'objectDetectionAP', 'normalizedMutualInformation',
        smaller_is_better = ['meanSquaredError', 'rootMeanSquaredError', 'rootMeanSquaredErrorAvg',
                             'meanAbsoluteError']
        # Smaller is better
        # 'meanSquaredError', 'rootMeanSquaredError', 'rootMeanSquaredErrorAvg',
        #'meanAbsoluteError'

        assert base['cross_validation_metrics'][0]["metric"] == \
               check['cross_validation_metrics'][0]["metric"], "cross_validation_metrics not equal"
        assert base['training_metrics'][0]["metric"] == \
               check['training_metrics'][0]["metric"], "training_metrics not equal"
        assert base['test_metrics'][0]["metric"] == \
               check['test_metrics'][0]["metric"], "test_metrics not equal"

        opr = lambda a,b: False
        if check['cross_validation_metrics'][0]["metric"] in larger_is_better:
            opr = operator.gt
        elif check['cross_validation_metrics'][0]["metric"] in smaller_is_better:
            opr = operator.lt

        metric_list = ["cross_validation_metrics", "test_metrics"]
        pprint(['cross_validation_metrics'][0])
        comparison_results = list(map(lambda m: opr(check[m][0]['value'], base[m][0]['value']),
                                 metric_list))
        if operator.xor(comparison_results[0],comparison_results[1]):
            _logger.warning("[WARN] cross_validation_metrics and test_metrics are not compatible")
            print("[WARN] cross_validation_metrics:{}, test_metrics:{}".format(*comparison_results))

        return operator.and_(comparison_results[0], comparison_results[1])

