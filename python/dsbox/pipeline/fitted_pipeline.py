import typing
from d3m.metadata.pipeline import Pipeline
from d3m.runtime import Runtime

class FittedPipeline:
    """
    Fitted pipeline

    Attributes
    ----------
    pipeline: Pipeline
        a pipeline
    dataset_id: str
        identifier for a dataset
    runtime: Runtime
        runtime containing fitted primitives
    """
    def __init__(self, pipeline: Pipeline, *, dataset_id: str = None, runtime: typing.Optional[Runtime] = None) -> None:
        self.pipeline: Pipeline = pipeline
        self.dataset_id: str = dataset_id
        self.runtime: typing.Optional[Runtime] = runtime
