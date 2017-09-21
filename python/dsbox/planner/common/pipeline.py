import uuid
import copy

class Pipeline(object):
    """
    Defines a pipeline
    """

    def __init__(self, id=None, primitives=None):
        if id is None:
            id = str(uuid.uuid4())
        if primitives is None:
            primitives = []
        self.id = id
        self.primitives = primitives
        self.planner_result = None
        self.test_result = None

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

    def __str__(self):
        return str(self.primitives)

    def __repr__(self):
        return str(self.primitives)
