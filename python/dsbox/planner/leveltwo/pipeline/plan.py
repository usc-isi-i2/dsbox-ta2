class Plan (object):
    '''
    A Plan (pipeline) -- list of primitives
    '''
    INCOMPLETE = 1
    COMPLETE = 2
    EXECUTED = 3
    OPTIMIZED = 4
    FEATURIZED = 5

    def __init__(self):
        self.pipeline = []
        self.parameters = ()
        self.status = Plan.INCOMPLETE
        
    def getPrimitives(self):
        return self.pipeline
    
    def addPrimitive(self, primitive):
        self.pipeline.append(primitive)
        
    def insertPrimitive(self, primitive, before):
        self.pipeline.insert(self.pipeline.index(before), primitive)
        
    def setParameterValue(self, primitive, parameter, value):
        if not self.parameters[primitive.name]:
            self.parameters[primitive.name] = ()
        self.parameters[primitive.name][parameter] = value
        
    def getParameters(self, primitive):
        return self.parameters[primitive.name]
    
    def setStatus(self, status):
        self.status = status
    
    def getStatus(self):
        return self.status
    
        