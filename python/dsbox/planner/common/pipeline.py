import uuid
import copy
import threading

class PipelineExecutionResult(object):
    """
    Defines a pipeline execution result
    """
    def __init__(self, predictions, metric_values):
        self.predictions = predictions # Predictions dataframe
        self.metric_values = metric_values # Dictionary of metric to value

class Pipeline(object):
    """
    Defines a pipeline
    """
    def __init__(self, id=None, primitives=None, ensemble= None):
        if id is None:
            id = str(uuid.uuid4())

        if primitives is None:
            primitives = []
        self.id = id
        self.primitives = primitives

        # Execution Results
        self.planner_result = None
        self.test_result = None

        # Ensemble?
        self.ensemble = ensemble


        # Change notification
        self.changes = threading.Condition()
        self.finished = False

    def clone(self, idcopy=False):
        pipeline = copy.deepcopy(self)
        if not idcopy:
            pipeline.id = str(uuid.uuid4())
        return pipeline

    def setPipelineId(self, id):
        self.id = id

    def setPrimitives(self, primitives):
        self.primitives = primitives

    def addPrimitive(self, primitive):
        self.primitives.append(primitive)

    def length(self):
        return len(self.primitives)

    def getPrimitiveAt(self, index):
        return self.primitives[index]

    def insertPrimitiveAt(self, index, primitive):
        self.primitives.insert(index, primitive)

    def replacePrimitiveAt(self, index, primitive):
        self.primitives[index] = primitive

    def replaceSubpipelineAt(self, index, subpipeline):
        self.primitives[index:index] = subpipeline.primitives

    def notifyChanges(self):
        self.changes.acquire()
        self.changes.notifyAll()
        self.changes.release()

    def waitForChanges(self):
        self.changes.acquire()
        self.changes.wait()
        self.changes.release()

    def __str__(self):
        if self.ensemble:
            return str(self.ensemble.pipelines)
        else:
            return str(self.primitives)

    def __repr__(self):
        if self.ensemble:
            return str(self.ensemble.pipelines)
        else:
            return str(self.primitives)

    def __getstate__(self):
        return (self.id, self.primitives, self.planner_result, self.test_result, self.finished, self.ensemble)

    def __setstate__(self, state):
        self.id, self.primitives, self.planner_result, self.test_result, self.finished, self.ensemble= state
        self.changes = threading.Condition()
