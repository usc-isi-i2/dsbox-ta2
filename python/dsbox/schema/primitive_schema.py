"""D3M Primitive Annotation Schema 1.0
"""

from enum import Enum
import uuid

class PrimitiveAnnotationSchema(object):
    """Primitive Annotation Schema version 1.0"""
    def __init__(self):
        self.name = ''
        self.common_name = ''
        self.description = ''

        self.languages = []  # string[]
        self.library = ''
        self.version = ''
        self.source_code = ''
        self.is_class = False
        self.attributes = []  # List of attributes of this primitive (see attribute schema below)
        self.parameters = []  # List of parameter of this primitive (see parameter schema below)
        self.methods_available = []  # A list of all methods available in this primitive (see method schema below)
        self.algorithm_type = []  # A list of algorithms implemented by this primitive (refer below)
        self.learning_type = []  # String[] The categories of learning (supervised, unsupervised, etc) this primitive may fall into (refer below)
        self.task_type = []  # String[] The types of task achieved by this primitive, e.g. - modeling, evaluation, etc (refer below)
        self.tags = []  # String[] A list of user defined tags attached to this primitive
        self.is_deterministic = True  # True, if the primitive is deterministic
        self.handles_classification = False  # True, if the primitive handles classification
        self.handles_regression = False  # True, if the primitive handles regression
        self.handles_multiclass = False  # True, if the primitive handles multiclass classification
        self.handles_multilabel = False  # True, if the primitive handles multilabel classification
        self.input_type = []  # String[] The type of inputs accepted by this primitive e.g. - DENSE, SPARSE, etc (refer below)
        self.output_type = [] # String[] The type of outputs produced by this primitive, e.g. - PREDICTIONS, etc (refer below)
        self.team = 'USC ISI' # The name of the team submitting the primitive
        self.metadata = None  # Reserved for future use
        self.schema_version = 1.0  # The version of the annotation schema, current one is 1.0
        self.build = []  # List[Object] How to build the primitive software stack (see build schema below)
        self.compute_resources = None # ObjectyesAn estimate of the compute resources required to run the primitive (see compute resources schema below)
    @property
    def id(self):
        return uuid.uuid3(uuid.uuid3(uuid.NAMESPACE_DNS, "datadrivendiscovery.org"), self.name + self.version).hex

class AttributeSchema(object):
    def __init__(self):
        self.name = ''  # Name of the attribute
        self.description = ''  # Description of this attribute
        self.type = ''  # Type of this attribute
        self.optional = False  # A value representing whether this attribute is optional or not. The value maybe be true or false
        self.shape = ''  # Shape of this attribute

class MethodSchema(object):
    def __init__(self):
        self.id = ''  # Fully qualified name of this method
        self.name = ''  # Name of the method
        self.description = ''  # Description of this method
        self.parameters = []  # A list of parameters of this method
        self.returns = None  # Object The return value of this method

class ParameterSchema(object):
    def __init__(self):
        self.name = ''  # StringyesName of the parameter
        self.description = ''  # StringnoDescription of this parameter
        self.type = ''  # StringyesType of this parameter
        self.optional = False  # A value representing whether this parameter is optional or not. The value maybe be true or false
        self.default = ''  # StringnoA default value if any
        self.shape = ''  # Shape of this parameter
        self.size = ''  # Stringnosize of this parameter
        self.is_hyperparameter = False  # Whether this is a hyperparameter or not

class BuildSchema(object):
    def __init__(self):
        self.build = []

ALGORITHM_TYPE = [
    'Bayesian',
    'Clustering',
    'Decision Tree',
    'Deep Learning',
    'Dimensionality Reduction',
    'Ensemble',
    'Instance Based',
    'Neural Networks',
    'Regularization',
    'Probabilistic Graphical Models'
]

TASK_TYPE = [
    'Data preprocessing',
    'Feature extraction',
    'Modeling',
    'Evaluation'
]

LEARNING_TYPE = [
    'Reinforcement learning',
    'Semi-supervised learning',
    'Supervised learning',
    'Unsupervised learning'
]

INPUT_TYPE = [
    'DENSE',
    'SPARSE',
    'UNSIGNED_DATA'
]

OUTPUT_TYPE = [
    'PREDICTIONS',
    'FEATURES'
]
