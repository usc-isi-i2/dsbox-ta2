class Primitive(object):
    """
    Defines a primitive and its details
    """

    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.task = None
        self.type = None
        self.preconditions = {}
        self.effects = {}
        self.column_primitive = False
        self.init_args = []
        self.init_kwargs = {}

    def addPrecondition(self, precondition):
        self.preconditions = dict(self.preconditions, **precondition)

    def addEffect(self, effect):
        self.effects = dict(self.effects, **effect)

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
