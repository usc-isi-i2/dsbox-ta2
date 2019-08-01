
class PipelineInstantiationError(RuntimeError):
    '''
    Failed to create pipeline from template
    '''

class PipelineEvaluationError(RuntimeError):
    '''
    Failed to run fit/produce on pipeline.
    '''

class PipelinePickleError(RuntimeError):
    '''
    Pipeline failed to properly pickle/unpickle
    '''

# class MetricsMismatchError(ValueError):
#     '''
#     Mismatch between metrics generate by original and pickled pipelines.
#     '''
