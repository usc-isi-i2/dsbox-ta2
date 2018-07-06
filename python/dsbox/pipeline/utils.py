

def larger_is_better(metric: str) -> bool:
    return not ('Error' in metric)

    # Larger is better
    # 'accuracy', 'precision', 'recall', 'f1', 'f1Micro', 'f1Macro', 'rocAuc',
    # 'rocAucMicro', 'rocAucMacro', 'rSquared', 'jaccardSimilarityScore',
    # 'precisionAtTopK', 'objectDetectionAP', 'normalizedMutualInformation',

    # Smaller is better
    #'meanSquaredError', 'rootMeanSquaredError', 'rootMeanSquaredErrorAvg',
    #'meanAbsoluteError'
