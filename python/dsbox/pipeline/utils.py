import logging
logger = logging.getLogger(__name__)


def larger_is_better(metric: str) -> bool:
    if type(metric) is str:
        return not ('Error' in metric)
    elif type(metric) is dict and 'metric' in metric:
        return not ('Error' in metric['metric'])
    else:
        # use not correctly, but still return something to ensure the program finished
        print("larger_is_better used not correctly!")
        logger.error("larger_is_better used not correctly!")
        return True
    # Larger is better
    # 'accuracy', 'precision', 'recall', 'f1', 'f1Micro', 'f1Macro', 'rocAuc',
    # 'rocAucMicro', 'rocAucMacro', 'rSquared', 'jaccardSimilarityScore',
    # 'precisionAtTopK', 'objectDetectionAP', 'normalizedMutualInformation',

    # Smaller is better
    #'meanSquaredError', 'rootMeanSquaredError', 'rootMeanSquaredErrorAvg',
    #'meanAbsoluteError'
