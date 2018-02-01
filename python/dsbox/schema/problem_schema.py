"""D3M Problem Schema Version
"""

from enum import Enum

# D3M Problem Schema
# https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/blob/shared/schemas/problemSchema.json
# Updated 1/27/2018

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    LINK_PREDICTION = "linkPrediction"
    VERTEX_NOMINATION = "vertexNomination"
    COMMUNITY_DETECTION = "communityDetection"
    GRAPH_CLUSTERING = "graphClustering"
    GRAPH_MATCHING = "graphMatching"
    TIME_SERIES_FORECASTING = "timeSeriesForecasting"
    COLLABORATIVE_FILTERING = "collaborativeFiltering"
 
class TaskSubType(Enum):
    NONE = "none"
    BINARY = "binary"  # applicable for classification, vertexNomination
    MULTICLASS = "multiClass"  # applicable for classification, vertexNomination
    MULTILABEL = "multiLabel"  # applicable for classification
    UNIVARIATE = "univariate"  # applicable for regression
    MULTIVARIATE = "multivariate"  # applicable for regression
    OVERLAPPING = "overlapping"  # applicable for community detection
    NONOVERLAPPING = "nonOverlapping" # applicable for community detection

class Metric(Enum):
    ACCURACY = "accuracy"  #sklearn.metrics.accuracy_score
    F1 = "f1"  #sklearn.metrics.f1_score
    F1_MICRO = "f1Micro"  #sklearn.metrics.f1_score(average='micro')
    F1_MACRO = "f1Macro"  #sklearn.metrics.f1_score(average='macro')
    ROC_AUC = "rocAuc"  #sklearn.metrics.roc_auc_score
    ROC_AUC_MICRO = "rocAucMicro"  #sklearn.metrics.roc_auc_score(average='micro')
    ROC_AUC_MACRO = "rocAucMacro"  #sklearn.metrics.roc_auc_score(average='macro')
    MEAN_SQUARED_ERROR = "meanSquaredError"  #sklearn.metrics.mean_squared_error
    ROOT_MEAN_SQUARED_ERROR = "rootMeanSquaredError"  #sqrt(sklearn.metrics.mean_squared_error)
    ROOT_MEAN_SQUARED_ERROR_AVG = "rootMeanSquaredErrorAvg"  #sum(mean_squared_error_list)/len(mean_squared_error_list)
    MEAN_ABSOLUTE_ERROR = "meanAbsoluteError"  #sklearn.metrics.median_absolute_error
    R_SQUARED = "rSquared"  #sklearn.metrics.r2_score
    NORMALIZED_MUTUAL_INFORMATION = "normalizedMutualInformation"  # see nmi.py
    JACCARD_SIMILARITY_SCORE = "jaccardSimilarityScore" # see jaccard.py script
    PRECISION_AT_TOP_K = "precisionAtTopK"
    EXECUTION_TIME = "executionTime"
