class Primitive(object):
    """
    Defines a primitive and its details
    """

    def __init__(self, name, cls):
        # Basic information from D3M library
        self.name = name
        self.cls = cls
        self.task = None
        self.type = None

        # Extra information added by our custom library
        self.preconditions = {}
        self.effects = {}
        self.is_persistent = True
        self.column_primitive = False
        self.unified_interface = False
        self.init_args = []
        self.init_kwargs = {}

        # Execution details
        self.executables = {}
        self.start_time = None
        self.end_time = None
        self.progress = 0.0
        self.finished = False

        self.pipeline = None # The pipeline that this primitive is a part of (if any)

    def addPrecondition(self, precondition):
        self.preconditions.update(precondition)

    def addEffect(self, effect):
        self.effects.update(effect)

    def getPreconditions(self):
        return self.preconditions

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
