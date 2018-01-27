from d3m_metadata.metadata import PrimitiveMetadata

class Primitive(object):
    """
    Defines a primitive and its details
    """

    def __init__(self, pid, name, cls):
        # Basic information from D3M library
        self.id = pid  # Unique persistent id 
        self.name = name  
        self.cls : str = cls  # Python package path
        self.task = None
        self.type = None
        self.d3m_metadata : PrimitiveMetadata = None

        # Extra information added by our custom library
        self.preconditions = {}
        self.error_conditions = {}
        self.effects = {}
        self.is_persistent = True
        self.column_primitive = False
        self.unified_interface = False
        self.init_args = []
        self.init_kwargs = {}
        
        # planning related
        self.weight = 1

        # Execution details
        self.executables = {}
        self.start_time = None
        self.end_time = None
        self.progress = 0.0
        self.finished = False

        self.pipeline = None # The pipeline that this primitive is a part of (if any)

    def addPrecondition(self, precondition):
        self.preconditions.update(precondition)

    def addErrorCondition(self, condition):
        self.error_conditions.update(condition)

    def addEffect(self, effect):
        self.effects.update(effect)

    def getPreconditions(self):
        return self.preconditions
    
    def getErrorCondition(self):
        return self.error_conditions    

    def getEffects(self):
        return self.effects

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def getInitArgs(self):
        return self.init_args

    def getInitKeywordArgs(self):
        return self.init_kwargs

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    '''
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        self_dict['executables'] = {}
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
    '''
