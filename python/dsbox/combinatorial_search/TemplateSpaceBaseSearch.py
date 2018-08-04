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
from dsbox.combinatorial_search.cache import CacheManager
from dsbox.template.template import DSBoxTemplate

# from dsbox.template.pipeline_utilities import pipe2str
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

        self.bestResult = None
        self.best_20_results: typing.List = []
        self.cacheManager = CacheManager()

        # load libraries with a dummy evaluation
        self.confSpaceBaseSearch[0].dummy_evaluate()


    def search(self, num_iter=1):
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
            print(STYLE+"[INFO] Selecting Template:", search.template.template['name'])

            if self.cacheManager.candidate_cache.is_hit(candidate):
                report = self.cacheManager.candidate_cache.lookup(candidate)
                assert report is not None and 'configuration' in report, \
                    'invalid candidate_cache line: {}->{}'.format(candidate, report)
            else:
                try:
                    report = search.evaluate_pipeline(
                        args=(candidate, self.cacheManager.primitive_cache, True))
                    self._update_best_result(report)
                    self.cacheManager.candidate_cache.push(report)
                except:
                    traceback.print_exc()
                    print(ERROR + "[INFO] Search Failed on candidate")
                    pprint(candidate)
                    self.cacheManager.candidate_cache.push_None(candidate=candidate)

        self.cacheManager.cleanup()
        return self.bestResult

    def _update_best_result(self, new_res: typing.Dict) -> None:
        if new_res is None:
            return
        # pprint(new_res)
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

        opr = lambda a, b: False
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
            print("[WARN] cross_validation_metrics:{} and test_metrics:{} are not "
                  "compatible".format(*comparison_results))

        # return operator.and_(comparison_results[0], comparison_results[1])
        return comparison_results[0]

    def _get_best_candidates(self, num: int = 20) -> typing.List[typing.Dict]:
        pass