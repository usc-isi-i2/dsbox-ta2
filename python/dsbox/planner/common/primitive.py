import sys
from hashlib import blake2b
import pickle

from d3m.metadata.base import PrimitiveMetadata, PrimitiveFamily, PrimitiveAlgorithmType
from d3m.metadata.hyperparams import Hyperparams
from d3m.primitive_interfaces.base import PrimitiveBase

class Primitive(object):
    """
    Defines a primitive and its details
    """

    def __init__(self, pid, name, cls):
        # Basic information from D3M library
        self.id = pid  # Unique persistent id
        self.name = name
        self.cls: str = cls  # Python package path
        self.task = None
        self.type = None
        self.d3m_metadata: PrimitiveMetadata = None

        self._has_hyperparameter_class = None
        self._hyperparams_class = None
        self._hyperparams: Hyperparams = None
        self.using_default_hyperparams = False

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
        self.executable_size = 0
        self.start_time = None
        self.end_time = None
        self.progress = 0.0
        self.finished = False

        self.pipeline = None # The pipeline that this primitive is a part of (if any)

    def hasHyperparamClass(self):
        '''Returns True is primitive has hyperparameter class'''
        if self._has_hyperparameter_class is None:
            mod, cls = self.cls.rsplit('.', 1)
            try:
                import importlib
                module = importlib.import_module(mod)
                primitive_class = getattr(module, cls)
                if issubclass(primitive_class, PrimitiveBase):
                    self._hyperparams_class = primitive_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    self._has_hyperparameter_class = True
                else:
                    self._has_hyperparameter_class = False
            except Exception as e:
                sys.stderr.write("ERROR: cannot get hyperparameter class {}: {}\n".format(self.name, e))
                self._has_hyperparameter_class = False
        return self._has_hyperparameter_class

    def getHyperparamClass(self):
        '''Returns the hyperparameter class'''
        if self.hasHyperparamClass():
            return self._hyperparams_class
        return None

    def getHyperparams(self) -> Hyperparams:
        if self._hyperparams is None:
            mod, cls = self.cls.rsplit('.', 1)
            try:
                import importlib
                module = importlib.import_module(mod)
                primitive_class = getattr(module, cls)
                if issubclass(primitive_class, PrimitiveBase):
                    hyperparams_class = primitive_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    self._hyperparams = hyperparams_class.defaults()
                    self.using_default_hyperparams = True
                else:
                    self._hyperparams = None
            except Exception as e:
                sys.stderr.write("ERROR: instantiate_primitive {}: {}\n".format(self.name, e))
                #sys.stderr.write("ERROR _instantiate_primitive(%s)\n" % (primitive.name))
                #traceback.print_exc()
                self._hyperparams = None

        return self._hyperparams

    def setHyperparams(self, hyperparams: Hyperparams):
        hclass = self.getHyperparamClass()
        self.using_default_hyperparams = hclass.defaults == hyperparams
        self._hyperparams = hyperparams

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
        if self._hyperparams is None:
            return '{}:None'.format(self.name)
        hash = blake2b(digest_size=10)
        hash.update(str(self._hyperparams).encode())
        return '{}:{}'.format(self.name, hash.hexdigest())

    def __repr__(self):
        if self._hyperparams is None:
            return '{}:None'.format(self.name)
        hash = blake2b(digest_size=10)
        hash.update(str(self._hyperparams).encode())
        return '{}:{}'.format(self.name, hash.hexdigest())

    def getFamily(self) -> PrimitiveFamily:
        return self.d3m_metadata.query()['primitive_family']

    def getAlgorithmTypes(self) -> PrimitiveAlgorithmType:
        return self.d3m_metadata.query()['algorithm_types']

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        self_dict['d3m_metadata'] = None
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def getExecutableSize(self):
        if self.executable_size == 0:
            if self.executables:
                try:
                    self.executable_size = sys.getsizeof(pickle.dumps(self.executables))
                except Exception as e:
                    sys.stderr.write("ERROR: getExecutableSize {}: {}\n".format(self.name, e))
                    self.executable_size = sys.maxsize
        return self.executable_size
