class Primitive(object):
    """
    Defines a primitive and its details
    """

    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.task = None
        self.types = []
        self.preconditions = []
        self.effects = []
        self.column_primitive = False
        
    def addPrecondition(self, precondition):
        self.preconditions.append(precondition)
        
    def addEffect(self, effect):
        self.effects.append(effect)

    def getPreconditions(self):
        return self.preconditions
    
    def getEffects(self):
        return self.effects
    
    def getName(self):
        return self.name
    
    def getTypes(self):
        return self.types
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    