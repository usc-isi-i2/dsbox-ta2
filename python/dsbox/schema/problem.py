from enum import Enum

from d3m.metadata.problem import PerformanceMetric

class OptimizationType(Enum):
    MAXIMIZE = 0
    MINIMIZE = 1

def optimization_type(metric: PerformanceMetric) -> OptimizationType:
    if metric in [PerformanceMetric.MEAN_SQUARED_ERROR,
                  PerformanceMetric.ROOT_MEAN_SQUARED_ERROR,
                  # PerformanceMetric.ROOT_MEAN_SQUARED_ERROR_AVG,
                  PerformanceMetric.MEAN_ABSOLUTE_ERROR,
                  PerformanceMetric.R_SQUARED]:
        return OptimizationType.MINIMIZE
    else:
        # PerformanceMetric.ACCURACY, PerformanceMetric.F1,
        # PerformanceMetric.F1_MICRO, PerformanceMetric.F1_MACRO,
        # PerformanceMetric.ROC_AUC, PerformanceMetric.ROC_AUC_MICRO,
        # PerformanceMetric.ROC_AUC_MACRO,
        # PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION,
        # PerformanceMetric.JACCARD_SIMILARITY_SCORE,
        # PerformanceMetric.PRECISION_AT_TOP_K
        return OptimizationType.MAXIMIZE
 


