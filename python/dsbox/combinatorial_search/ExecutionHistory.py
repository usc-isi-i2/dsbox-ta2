import pandas as pd
import typing
import operator
import logging
import copy
import math
from pprint import pprint
from functools import reduce

_logger = logging.getLogger(__name__)


class ExecutionHistory:
    comparison_metrics = ['training_metrics', 'cross_validation_metrics', 'test_metrics']

    """
    report = {
                'fitted_pipeline': ,
                'training_metrics': [{"metric": , "value": }],
                'cross_validation_metrics': [{"metric": , "value": }],
                'test_metrics': [{"metric": , "value": }],
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
                'sim_count': int ????
            }
    """

    def __init__(self, template_list: typing.List = None,
                 key_attribute: str = 'cross_validation_metrics'):

        self.total_run = 0
        self.total_time = 0

        template_names = map(lambda s: s.template["name"], template_list) if template_list else \
            ['generic']

        self.storage = pd.DataFrame(
            None,
            index=template_names,
            columns=['training_metrics', 'cross_validation_metrics', 'test_metrics',
                     'sim_count', 'total_runtime', 'configuration'])
        self.storage[['total_runtime', 'sim_count']] = 0

        # setup configuration field
        self.storage['configuration'] = self.storage['configuration'].astype(object)
        self.storage['configuration'] = None

        self.key_attribute = key_attribute

    def update(self, report: typing.Dict, template_name: str = 'generic') -> None:
        """
        updates the execution history based on the report input. The report dict should be of the
        format as indicated below.
        Examples
            r = {
                'fitted_pipeline': ,
                'training_metrics': [{"metric": , "value": }],
                'cross_validation_metrics': [{"metric": , "value": }],
                'test_metrics': [{"metric": , "value": }],
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
                'sim_count': int ????
            }
            e = ExecutionHistory(...)
            e.update(template_name="...",report=r)
        Args:
            template_name: str
                name of the template
            report: typing.Dict
                the report dictionary

        Returns:
            None
        """
        if 'sim_count' not in report:
            report['sim_count'] = 1
        # update the global statistics
        self.total_run += report['sim_count']
        self.total_time += report['total_runtime']

        row = self.storage.loc[template_name]

        # these fields will be updated no matter what is the result
        update = {
            'sim_count': row['sim_count'] + report['sim_count'],
            'total_runtime': row['total_runtime'] + report['total_runtime'],
        }

        # these fields will be updated only if the new report is better than the previous ones
        if ExecutionHistory._is_better(base=row, check=report, key_attribute=self.key_attribute):
            for k in ['configuration', 'training_metrics', 'cross_validation_metrics',
                      'test_metrics']:
                update[k] = report[k]

        # apply all the changes
        for k in update:
            self.storage.loc[template_name][k] = update[k]

    def normalize(self) -> pd.DataFrame:
        """
        Returns the normalized version of execution history. The normalized dataframe only
        contains the value not other metadata related to each one of the attibutes of history (
        e.g. evaluation metric's names)

        Returns:
            normalize: pd.DataFrame
                normalized history dataframe
        """
        alpha = 0.01
        normalize = pd.DataFrame(
            0,
            index=self.storage.index.tolist(),
            columns=['training_metrics', 'cross_validation_metrics', 'test_metrics',
                     'sim_count', 'total_runtime'])

        for k in ['training_metrics', 'cross_validation_metrics', 'test_metrics']:
            normalize[k] = self.storage[k][0]['value']

        for k in ['sim_count', 'total_runtime']:
            normalize[k] = self.storage[k]

        scale = (normalize.max() - normalize.min())
        scale.replace(to_replace=0, value=1, inplace=True)
        normalize = (normalize - normalize.min()) / scale
        normalize.clip(lower=0.01, upper=1, inplace=True)

        return normalize

    def update_none(self, fail_report: typing.Dict, template_name: str = 'generic') -> None:

        if fail_report is None:
            fail_report = {}
        if 'sim_count' not in fail_report:
            fail_report['sim_count'] = 1
        if 'total_runtime' not in fail_report:
            fail_report['total_runtime'] = 0

        fail_report['Error'] = 1
        self.update(report=fail_report, template_name=template_name)
        return

    @staticmethod
    def _is_better(base: typing.Dict, check: typing.Dict, key_attribute: str) -> bool:
        if 'Error' in check:
            return False

        if base is None:
            return True
        # check if the base contains all the metrics. If they are not valid it is either due to
        # uninitialized base (first update) or it is due to incomplete evaluation in previous
        # runs. The code will favor the most complete one between check and base.
        check_comparison_metrics = copy.deepcopy(ExecutionHistory.comparison_metrics)
        base_comparison_metrics = copy.deepcopy(ExecutionHistory.comparison_metrics)
        for s in ExecutionHistory.comparison_metrics:
            if base[s] is None or (isinstance(base[s], float) and math.isnan(base[s])):
                base_comparison_metrics.remove(s)
            if s not in check or check[s] is None:
                check_comparison_metrics.remove(s)

        # check if base is uninitialized
        if len(base_comparison_metrics) == 0:
            return True

        # FIXME at this point we favor the most complete one. Maybe we need to change this
        if len(base_comparison_metrics) != len(check_comparison_metrics):
            return len(base_comparison_metrics) < len(check_comparison_metrics)
            # true if check is more complete and false if base is the more complete

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
        # 'meanAbsoluteError'

        comparison_results = {}
        for s in base_comparison_metrics:
            assert base[s][0]["metric"] == check[s][0]["metric"], "{} not equal".format(s)

            opr = lambda a, b: False
            if check[s][0]["metric"] in larger_is_better:
                opr = operator.gt
            elif check[s][0]["metric"] in smaller_is_better:
                opr = operator.lt

            comparison_results[s] = opr(check[s][0]['value'], base[s][0]['value'])

        if len(set(comparison_results.values())) != 1:
            _logger.warning("[WARN] cross_validation_metrics and test_metrics are not compatible")
            print(("[WARN]" + "{}:{}" * len(comparison_results) + "are not compatible").format(
                *[item for tup in zip(base_comparison_metrics, comparison_results) for item in
                  tup]))

        # return operator.and_(comparison_results[0], comparison_results[1])
        return comparison_results[key_attribute]

    def get_best_history(self) -> typing.List[typing.Dict]:
        best = None
        for t_name, row in self.storage.iterrows():
            if ExecutionHistory._is_better(base=best, check=row, key_attribute=self.key_attribute):
                best = row
        return best.to_dict()
