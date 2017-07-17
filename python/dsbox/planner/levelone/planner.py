"""This module implements the level one planner of the DSBox system.
"""

import json
import pkgutil
import random
import numpy as np

class Ontology(object):
    """Primitve ontology"""
    ONTOLOGY_FILE = 'ontology.json'

    def __init__(self):
        self.load()

    def load(self):
        """Load ontology from JSON definition"""
        text = pkgutil.get_data('dsbox.planner.levelone', self.ONTOLOGY_FILE)
        print(type(text))
        content = json.loads(text.decode())
        self.task = content['TaskOntology']
        self.learning_type = content['LearningTypeOntology']
        self.algo = content['MachineLearningAlgorithmOntology']

    def get_tasks(self):
        """Returns task names"""
        return [node["Name"] for node in self.task]

class Primitive(object):
    """A primitive"""
    def __init__(self, definition):
        self.name = definition['Name']
        self.task = definition['Task']
        self.learning_type = definition['LearningType']
        self.ml_algorithm = definition['MachineLearningAlgorithm']

    def __str__(self):
        return 'Primitive({})'.format(self.name)

    def __eq__(self, other):
        """Define equals based on name"""
        if isinstance(other, self.__class__):
            return self.name == other.name
        return NotImplemented

    def __ne__(self, other):
        """Overide non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Overide hash by using name attribute"""
        return hash(self.name)

class D3Primitive(Primitive):
    def __init__(self, definition):
        self.name = definition['common_name']

        self.task = ''
        if 'task_type' in definition:
            if 'feature extraction' in definition['task_type']:
                self.task = "FeatureExtraction"
            if 'data preprocessing' in definition['task_type']:
                self.task = "DataPreprocessing"

        self.learning_type = 'NA'
        if 'handles_classification' in definition and definition['handles_classification']:
            self.learning_type = 'Classification'
        if 'handles_regression' in definition and definition['handles_regression']:
            self.learning_type = 'Regression'

        if 'algorithm_type' in definition:
            # !!!! For now get only the first type
            self.ml_algorithm = definition['algorithm_type'][0]
        else:
            self.ml_algorithm = 'NA'


class Primitives(object):
    """Maintain available primitives"""
    PRIMITIVE_FILE = 'primitives.json'

    def __init__(self):
        self._load()
        self._index = dict()
        for index, primitive in enumerate(self.primitives):
            self._index[primitive.name] = index
        self.size = len(self.primitives)

    def _load(self):
        """Load primitive definition from JSON file"""
        text = pkgutil.get_data('dsbox.planner.levelone', self.PRIMITIVE_FILE)
        content = json.loads(text.decode())
        self.primitives = [Primitive(primitive_dict) for primitive_dict in content['Primitives']]

    def filter_equality(self, aspect, name):
        """Find primitive by aspect and name value"""
        result = [p for p in self.primitives if getattr(p, aspect) == name]
        return result

    def filter_by_task(self, name):
        """Find primitive by task aspect and name value"""
        return self.filter_equality('task', name)

    def filter_by_learning_type(self, name):
        """Find primitive by learning-type aspect and name value"""
        return self.filter_equality('learning_type', name)

    def filter_by_algo(self, name):
        """Find primitive by algorithm aspect and name value"""
        return self.filter_equality('ml_algorithm', name)

    def get_by_name(self, name):
        """Get primitve by unique name"""
        for primitive in self.primitives:
            if primitive.name == name:
                return primitive
        return None

    def get_index(self, name):
        """Returns the index of the primitive given its name"""
        return self._index[name]

#    def get_supervised_learning_by_algorithm(self):
#        for learning_type in ['Classification', 'Regression']:

class D3Primitives(Primitives):
    PRIMITIVE_FILE = 'sklearn.json'
    def __init__(self):
        self._load()
        self._index = dict()
        for index, primitive in enumerate(self.primitives):
            self._index[primitive.name] = index
        self.size = len(self.primitives)

    def _load(self):
        text = pkgutil.get_data('dsbox.planner.levelone', self.PRIMITIVE_FILE)
        content = json.loads(text.decode())
        self.primitives = [D3Primitive(primitive_dict) for primitive_dict in content['search_primitives']]

class ConfigurationSpace(object):
    """Defines the space of primitive pipelines"""

    def __init__(self, dimension_names, space):
        assert len(dimension_names) == len(space)
        self.ndim = len(dimension_names)
        self.dimension_names = dimension_names
        self.space = space
        self._subspace_lookup = dict()
        for name, subspace in zip(dimension_names, space):
            self._subspace_lookup[name] = subspace

    @classmethod
    def set_seed(cls, a_seed):
        """Set random seed"""
        random.seed(a_seed)

    def get_random_configuration(self):
        """Returns a random configuration from the configuration space"""
        components = []
        for component_space in self.space:
            i = random.randrange(len(component_space))
            component = component_space[i]
            components.append(component)
        return ConfigurationPoint(self, components)

    def get_configuration_by_policy(self, policy):
        """Returns a random configuration based on stochastic policy"""
        components = []
        component_names = []
        for component_space in self.space:
            names = [p.name for p in component_space]
            if components:
                values = policy.get_affinities(component_names, names)
                rand = np.sum(values) * random.random()
                i = 0
                while rand > values[i]:
                    rand -= values[i]
                    i += 1
            else:
                i = random.randrange(len(component_space))
            component = component_space[i]
            components.append(component)
            component_names.append(component.name)
        return ConfigurationPoint(self, components)

class ConfigurationPoint(object):
    """A point in ConfigurationSpace"""
    def __init__(self, configuration_space, point=None):
        self.configuration_space = configuration_space
        if point is None:
            self.point = configuration_space.ndim * [None]
        else:
            self.set_point(point)

    def set_point(self, point):
        """set point value list"""
        assert len(point) == self.configuration_space.ndim
        for dim, subspace in zip(point, self.configuration_space.space):
            if dim is not None:
                assert dim in subspace
        self.point = point


class AffinityPolicy(object):
    """Defines affinity pairs of components.
    The default affinity is 1. If affinity value is 10 then the pair
    is 10x more likely to occur together.
    """
    def __init__(self, primitives):
        self.primitives = primitives
        self.affinity_matrix = np.ones((primitives.size, primitives.size))

    def set_affinity(self, source_primitive, dest_primitive, affinity_value):
        """Set affinity between source to destination primitive"""
        row = self.primitives.get_index(source_primitive)
        col = self.primitives.get_index(dest_primitive)
        self.affinity_matrix[row, col] = affinity_value

    def set_symetric_affinity(self, source_primitive, dest_primitive, affinity_value):
        """Set affinities between the two primitives"""
        row = self.primitives.get_index(source_primitive)
        col = self.primitives.get_index(dest_primitive)
        self.affinity_matrix[row, col] = affinity_value
        self.affinity_matrix[col, row] = affinity_value

    def get_affinity(self, source_primitive, dest_primitive):
        """Returns affinity from source to desintation primitive.
        Default value is zero."""
        row = self.primitives.get_index(source_primitive)
        col = self.primitives.get_index(dest_primitive)
        return self.affinity_matrix[row, col]

    def get_affinities(self, source_primitives, dest_primitives):
        """Returns vector of affinitites of length len(dest_primitives)"""
        source_index = [self.primitives.get_index(p) for p in source_primitives]
        dest_index = [self.primitives.get_index(p) for p in dest_primitives]
        result = np.zeros(len(dest_primitives))
        for pos, dst in enumerate(dest_index):
            affinity_sum = 0
            for src in source_index:
                affinity_sum += self.affinity_matrix[src, dst]
            result[pos] = affinity_sum
        return result

class Pipeline(object):
    """Defines a sequence of executions"""
    def __init__(self, configuration_point):
        self.configuration_point = configuration_point

    @classmethod
    def get_random_pipeline(cls, configuration_space):
        """Returns a random pipeline"""
        return Pipeline(configuration_space.get_random_configuration())

    def __str__(self):
        out_list = []
        for name, c in zip(self.configuration_point.configuration_space.dimension_names,
                           self.configuration_point.point):
            if c is None:
                out_list.append('{}=None'.format(name))
            else:
                out_list.append('{}={}'.format(name,c.name))
        return 'Pipeline(' + ', '.join(out_list) + ')'

class LevelOnePlanner(object):
    """Level One Planner"""
    def __init__(self, learning_type='Classification', evaluator_name='F1',
                 primitives=None):
        self.learning_type = learning_type
        self.evaluator_name = evaluator_name
        if primitives:
            self.primitives = primitives
        else:
            self.primitives = Primitives()

    def generate_pipelines(self, num_pipelines=5, ignore_preprocessing=True):
        """Generation pipelines"""

        dimension_name = []
        dimension = []
        if not ignore_preprocessing:
            dimension_name.append('DataPreprocessing')
            preprocess = self.primitives.filter_by_task('DataPreprocessing')
            dimension.append(preprocess)

        dimension_name.append('FeatureExtraction')
        feature = self.primitives.filter_by_task('FeatureExtraction')
        dimension.append(feature)

        dimension_name.append(self.learning_type)
        learner = self.primitives.filter_by_learning_type(self.learning_type)
        dimension.append(learner)

        # algos = [(l['MachineLearningAlgorithm'], l) for l in learner]

        dimension_name.append('Evaluation')
        evaluator = [self.primitives.get_by_name(self.evaluator_name)]
        dimension.append(evaluator)

        # pprint.pprint(preprocess)
        # pprint.pprint(feature)
        # pprint.pprint(learner)
        # pprint.pprint(evaluator)

        configuration_space = ConfigurationSpace(dimension_name, dimension)

        pipelines = []
        for _ in range(num_pipelines):
            pipeline = Pipeline.get_random_pipeline(configuration_space)
            pipelines.append(pipeline)

        return pipelines

    def generate_pipelines_with_policy(self, policy, num_pipelines=5,
                                       ignore_preprocessing=True):
        """Generation pipelines using affinity policy"""

        dimension_name = []
        dimension = []
        if not ignore_preprocessing:
            dimension_name.append('DataPreprocessing')
            preprocess = self.primitives.filter_by_task('DataPreprocessing')
            dimension.append(preprocess)

        dimension_name.append('FeatureExtraction')
        feature = self.primitives.filter_by_task('FeatureExtraction')
        dimension.append(feature)

        dimension_name.append(self.learning_type)
        learner = self.primitives.filter_by_learning_type(self.learning_type)
        dimension.append(learner)

        dimension_name.append('Evaluation')
        evaluator = [self.primitives.get_by_name(self.evaluator_name)]
        dimension.append(evaluator)

        configuration_space = ConfigurationSpace(dimension_name, dimension)

        pipelines = []
        for _ in range(num_pipelines):
            configuration = configuration_space.get_configuration_by_policy(policy)
            pipeline = Pipeline(configuration)
            pipelines.append(pipeline)

        return pipelines

def main():
    """Test method"""
    primitives = Primitives()
    planner = LevelOnePlanner(primitives=primitives)
    policy = AffinityPolicy(primitives)

    policy.set_symetric_affinity('Descritization', 'NaiveBayes', 10)
    policy.set_symetric_affinity('Normalization', 'SVM', 10)


    pipelines = planner.generate_pipelines_with_policy(policy, 20,
                                                       ignore_preprocessing=True)
    for pipeline in pipelines:
        print(pipeline)

def testd3():
    """Test method"""
    primitives = D3Primitives()
    planner = LevelOnePlanner(primitives=primitives)
    policy = AffinityPolicy(primitives)

    pipelines = planner.generate_pipelines(20, ignore_preprocessing=True)
    for pipeline in pipelines:
        print(pipeline)

if __name__ == "__main__":
    main()
