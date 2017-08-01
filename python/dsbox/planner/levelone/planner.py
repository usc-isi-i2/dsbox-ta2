"""This module implements the level one planner of the DSBox system.
"""

from collections import defaultdict
from enum import Enum

import json
import pkgutil
import random
import pprint

import numpy as np

def random_choices(population, weights):
    """Randomly select a element based on weights. Same as random.choices in Python 3.6+"""
    assert len(population) == len(weights)
    total_weight = np.sum(weights)
    rand = total_weight * random.random()
    i = 0
    while rand > weights[i]:
        rand -= weights[i]
        i += 1
    return population[i]

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
    def __init__(self):
        self.name = ''
        self.task = ''
        self.learning_type = ''
        self.ml_algorithm = ''
        self.tags = ['NA', 'NA']
        self.weight = 1

    def __str__(self):
        return 'Primitive("{}")'.format(self.name)
    def __repr__(self):
        return 'Primitive("{}")'.format(self.name)

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

class DSBoxPrimitive(Primitive):
    """A primitive"""
    def __init__(self, definition):
        super().__init__()
        self.name = definition['Name']
        self.task = definition['Task']
        self.learning_type = definition['LearningType']
        self.ml_algorithm = definition['MachineLearningAlgorithm']
        self.tags = [self.ml_algorithm, self.ml_algorithm]
        self.weight = 1
    def __str__(self):
        return 'DSBoxPrimitive("{}")'.format(self.name)
    def __repr__(self):
        return 'DSBoxPrimitive("{}")'.format(self.name)

class D3mPrimitive(Primitive):
    """Primitive defined using D3M metadata"""
    def __init__(self, definition):
        super().__init__()
        self.name = definition['id'].split('.')[-1]

        self.task = ''
        if 'task_type' in definition:
            if 'feature extraction' in definition['task_type']:
                self.task = "FeatureExtraction"
            if 'data preprocessing' in definition['task_type']:
                self.task = "DataPreprocessing"

        self.learning_type = 'NA'
        if 'handles_classification' in definition and definition['handles_classification']:
            self.learning_type = 'Classification'
            self.task = 'Modeling'
        if 'handles_regression' in definition and definition['handles_regression']:
            self.learning_type = 'Regression'
            self.task = 'Modeling'

        if 'algorithm_type' in definition:
            # !!!! For now get only the first type
            self.ml_algorithm = definition['algorithm_type'][0]
        else:
            self.ml_algorithm = 'NA'

        self.tags = definition['tags']

        # make sure tag hierarchy is at least 2
        if len(self.tags) == 0:
            self.tags = ['NA', 'NA']
        elif len(self.tags) == 1:
            self.tags = [self.tags[0], self.tags[0]]
        self.weight = 1

    def __str__(self):
        return 'D3mPrimitive("{}")'.format(self.name)
    def __repr__(self):
        return 'D3mPrimitive("{}")'.format(self.name)

class HierarchyNode(object):
    """Node in the Hierarchy"""
    def __init__(self, hierarchy, name, content=None):
        self.hierarchy = hierarchy
        self.name = name
        self.children = []
        self.content = content
    def add_child(self, name, content=None):
        """Add a child to the hierarchy"""
        child = HierarchyNode(self.hierarchy, name, content)
        self.children.append(child)
        return child
    def has_child(self, name):
        """Return true if node has child with given name"""
        for node in self.children:
            if node.name == name:
                return True
        return False
    def get_child(self, name):
        """Return child by name"""
        for node in self.children:
            if node.name == name:
                return node
        raise Exception('Child not found: {}'.format(name))
    def __str__(self):
        return 'Node({},num_child={})'.format(self.name, len(self.children))


class Hierarchy(object):
    """Generic tree of nodes"""
    def __init__(self, name):
        # name of this hierarchy
        self.name = name
        self.root = HierarchyNode(self, 'root')
        self._changed = False
        self._level_count = []

    def add_child(self, node, name, content=None):
        """Create and add child node"""
        assert node.hierarchy == self
        node.add_child(name, content)
        self._changed = True

    def add_path(self, names):
        """Create and add all nodes in path"""
        curr_node = self.root
        for name in names:
            if curr_node.has_child(name):
                curr_node = curr_node.get_child(name)
            else:
                curr_node = curr_node.add_child(name)
                self._changed = True
        return curr_node

    def get_level_counts(self):
        """Computes the number of nodes at each level"""
        if not self._changed:
            return self._level_count
        self._level_count = self._compute_level_counts(self.root, 0, list())
        self._changed = False
        return self._level_count

    def _compute_level_counts(self, node, level, counts):
        """Computes the number of nodes at each level"""
        if len(counts) < level + 1:
            counts = counts + [0]
        counts[level] += 1
        for child in node.children:
            counts = self._compute_level_counts(child, level+1, counts)
        return counts

    def get_primitive_count(self):
        """Returns the number of primitives"""
        return self._get_primitive_count(self.root)

    def _get_primitive_count(self, curr_node):
        """Returns the number of primitives"""
        count = 0
        if curr_node.content is not None:
            count += len(curr_node.content)
        for child in curr_node.children:
            count += self._get_primitive_count(child)
        return count

    def get_primitives(self, curr_node=None):
        if curr_node is None:
            curr_node = self.root
        result = []
        if curr_node.content is not None:
            result.append(curr_node.content)
        for child in curr_node.children:
            result += self.get_primitives(child)
        return result

    def get_nodes_by_level(self, level):
        """Returns node at a specified level of the tree"""
        return self._get_nodes_by_level(self.root, 0, level)

    def _get_nodes_by_level(self, curr_node, curr_level, target_level):
        """Returns node at a specified level of the tree"""
        if curr_level >= target_level:
            return [curr_node]
        elif curr_level +1 == target_level:
            return curr_node.children
        else:
            result = []
            for node in curr_node.children:
                result += self._get_nodes_by_level(node, curr_level + 1, target_level)
            return result

    def print(self):
        """Print hierarchy"""
        print('Hierarchy({}, level_counts={})'.format(self.name, self.get_level_counts()))
        self._print(self.root, [])

    def _print(self, curr_node, path, max_depth=2):
        new_path = path + [curr_node]
        if len(new_path) > max_depth:
            print(' '*4 + ':'.join([node.name for node in new_path[1:]]))
            for line in pprint.pformat(curr_node.content).splitlines():
                print(' '*8 + line)
        else:
            for child in curr_node.children:
                self._print(child, new_path, max_depth=max_depth)

    def __str__(self):
        return 'Hierarchy({}, num_primitives={}, level_node_counts={})'.format(
            self.name, self.get_primitive_count(), self.get_level_counts())

class Category(Enum):
    PREPROCESSING = 1
    FEATURE = 2
    CLASSIFICATION = 3
    REGRESSION = 4
    UNSUPERVISED = 5
    EVALUATION = 6
    OTHER = 7

class Primitives(object):
    """Base Primitives class"""
    def __init__(self):
        self.primitives = []
        self._index = dict()
        self.size = 0
        self.hierarchy_types = [Category.PREPROCESSING, Category.FEATURE,
                                Category.CLASSIFICATION, Category.REGRESSION,
                                Category.UNSUPERVISED, Category.EVALUATION,
                                Category.OTHER]
        self.hierarchies = dict()
        for name in Category:
            self.hierarchies[name] = Hierarchy(name)

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

    def get_hierarchies(self):
        """Returns all primitive hierarchies as dict"""
        return self.hierarchies

    def print_statistics_old(self):
        """Print statistics of the primitives"""
        classification = 0
        regression = 0
        classification_algo = defaultdict(int)
        regression_algo = defaultdict(int)
        tag_primitive = defaultdict(list)
        for primitive in self.primitives:
            if len(primitive.tags) > 0:
                tag_str = ':'.join(primitive.tags)
            if primitive.learning_type == Category.CLASSIFICATION:
                classification += 1
                # classification_algo[primitive.ml_algorithm] += 1
                classification_algo[tag_str] += 1
                tag_primitive['C:' + tag_str].append(primitive.name)
            elif primitive.learning_type == Category.REGRESSION:
                regression += 1
                regression_algo[tag_str] += 1
                tag_primitive['R:' + tag_str].append(primitive.name)
            else:
                tag_primitive['O:' + tag_str].append(primitive.name)
        print('Primtive by Tag:')
        pprint.pprint(tag_primitive)
        print('Total number of primitives = {}'.format(self.size))
        print('num classifiers = {}'.format(classification))
        pprint.pprint(classification_algo)
        print('num regressors = {}'.format(regression))
        pprint.pprint(regression_algo)

    def print_statistics(self):
        """Print statistics of the primitives"""
        print('Total number of primitives = {}'.format(self.size))
        print('Number of primitives by hierarchy:')
        hierarchies = self.get_hierarchies()
        for name in Category:
            print(' '*4 + str(hierarchies[name]))

    def _compute_tag_hierarchy(self, classifer=[], regressor=[]):
        """Compute hierarchy based on sklearn tags"""
        for primitive in self.primitives:
            # Put base/mixin and functions into other category
            if (primitive.tags[0] == 'base'
                or (primitive.tags[1] == 'base'
                    and not 'LinearRegression' in primitive.name)
                or 'Base' in primitive.name
                or 'Mixin' in primitive.name
                or primitive.name[0].islower()
                or primitive.name == 'ForestRegressor'
                or primitive.name == 'ForestClassifier'):

                node = self.hierarchies[Category.OTHER].add_path(primitive.tags[:2])

            elif (primitive.learning_type == 'Classification'
                  or primitive.tags[0] in ['lda', 'qda', 'naive_bayes']
                  or ('Classifier' in primitive.name
                      and not primitive.tags[0] in
                      ['multiclass', 'multioutput', 'calibration'])
                  or 'SVC' in primitive.name
                  or 'LogisticRegression' in primitive.name
                  or 'Perceptron' in primitive.name  # Same as SGDClassifier
            ):

                node = self.hierarchies[Category.CLASSIFICATION].add_path(primitive.tags[:2])

                # Modify primitive learning type
                primitive.learning_type = 'Classification'

            elif (primitive.learning_type == 'Regression'
                  or primitive.tags[0] in ['isotonic']
                  or ('Regressor' in primitive.name
                      and not primitive.tags[0] in ['multioutput'])
                  or 'SVR' in primitive.name
                  or 'ElasticNet' in primitive.name
                  or 'KernelRidge' in primitive.name
                  or 'Lars' in primitive.name
                  or 'Lasso' in primitive.name
                  or 'LinearRegression' in primitive.name
                  or 'Ridge' in primitive.name):

                node = self.hierarchies[Category.REGRESSION].add_path(primitive.tags[:2])

                # Modify primitive learning type
                primitive.learning_type = 'Regression'

            elif primitive.tags[0] in ['cluster', 'mixture']:

                node = self.hierarchies[Category.UNSUPERVISED].add_path(primitive.tags[:2])

            elif (primitive.tags[0] in ['feature_extraction', 'feature_selection', 'decomposition',
                                       'random_projection', 'manifold']
                  or 'OrthogonalMatchingPursuit' in primitive.name):

                node = self.hierarchies[Category.FEATURE].add_path(primitive.tags[:2])

            elif primitive.tags[0] == 'preprocessing':

                node = self.hierarchies[Category.PREPROCESSING].add_path(primitive.tags[:2])

            elif (primitive.tags[0] in ['metrics', 'cross_validation', 'model_selection']):

                node = self.hierarchies[Category.EVALUATION].add_path(primitive.tags[:2])

            else:
                node = self.hierarchies[Category.OTHER].add_path(primitive.tags[:2])

            if node.content is None:
                node.content = [primitive]
            else:
                node.content.append(primitive)

class DSBoxPrimitives(Primitives):
    """Maintain available primitives"""
    PRIMITIVE_FILE = 'primitives.json'

    def __init__(self):
        super().__init__()
        self._load()
        for index, primitive in enumerate(self.primitives):
            self._index[primitive.name] = index
        self.size = len(self.primitives)
        self._compute_tag_hierarchy()

    def _load(self):
        """Load primitive definition from JSON file"""
        text = pkgutil.get_data('dsbox.planner.levelone', self.PRIMITIVE_FILE)
        content = json.loads(text.decode())
        self.primitives = [DSBoxPrimitive(primitive_dict)
                           for primitive_dict in content['Primitives']]

class D3mPrimitives(Primitives):
    """Primitives from D3M metadata"""

    PRIMITIVE_FILE = 'sklearn.json'
    def __init__(self, classifer=[], regressor=[]):
        super().__init__()
        self._load()
        for index, primitive in enumerate(self.primitives):
            self._index[primitive.name] = index
        self.size = len(self.primitives)
        self._compute_tag_hierarchy(classifer=classifer, regressor=regressor)

    def _load(self):
        """Load primitve from json"""
        text = pkgutil.get_data('dsbox.planner.levelone', self.PRIMITIVE_FILE)
        content = json.loads(text.decode())
        self.primitives = [D3mPrimitive(primitive_dict)
                           for primitive_dict in content['search_primitives']]


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
                component = random_choices(component_space, values)
            else:
                i = random.randrange(len(component_space))
                component = component_space[i]
            components.append(component)
            component_names.append(component.name)
        return ConfigurationPoint(self, components)

    def get_configuration_point(self, name_list, component_list):
        """Generate point from partially specified components"""
        for name in name_list:
            if not name in self.dimension_names:
                raise Exception('Invalid configuration dimension: {}'.format(name))
        components = []
        for name in self.dimension_names:
            if name in name_list:
                index = name_list.index(name)
                components.append(component_list[index])
            else:
                components.append(None)
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
        for name, component in zip(self.configuration_point.configuration_space.dimension_names,
                                   self.configuration_point.point):
            if component is None:
                out_list.append('{}=None'.format(name))
            else:
                out_list.append('{}={}'.format(name, component.name))
        return 'Pipeline(' + ', '.join(out_list) + ')'

class LevelOnePlanner(object):
    """Level One Planner"""
    def __init__(self, learning_type='Classification', evaluator_name='F1',
                 ignore_preprocessing=True, primitives=None):
        self.learning_type = learning_type
        self.evaluator_name = evaluator_name
        self.ignore_preprocessing = ignore_preprocessing
        if primitives:
            self.primitives = primitives
        else:
            self.primitives = Primitives()
        self.configuration_space = self.compute_configuration_space()

    def compute_configuration_space(self):
        """Compute configuration space using Primitives"""
        dimension_name = []
        dimension = []
        if not self.ignore_preprocessing:
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

        return ConfigurationSpace(dimension_name, dimension)

    def generate_pipelines(self, num_pipelines=5):
        """Generation pipelines"""

        pipelines = []
        for _ in range(num_pipelines):
            pipeline = Pipeline.get_random_pipeline(self.configuration_space)
            pipelines.append(pipeline)

        return pipelines

    def generate_pipelines_with_policy(self, policy, num_pipelines=5):
        """Generate pipelines using affinity policy"""

        pipelines = []
        for _ in range(num_pipelines):
            configuration = self.configuration_space.get_configuration_by_policy(policy)
            pipeline = Pipeline(configuration)
            pipelines.append(pipeline)

        return pipelines

    def generate_pipelines_with_hierarchy(self, level=2):
        """Generate singleton pipeline using tag hierarchy"""
        if self.learning_type == 'Classification':
            hierarchy = self.primitives.hierarchies[Category.CLASSIFICATION]
        elif self.learning_type == 'Regression':
            hierarchy = self.primitives.hierarchies[Category.REGRESSION]
        else:
            raise Exception('Learning type not recoginized: {}'.format(self.learning_type))
        if level == 1:
            nodes = hierarchy.get_nodes_by_level(1)
            primitives_by_node = []
            for l1_node in nodes:
                result = []
                for l2_node in l1_node.children:
                    result += l2_node.content
                primitives_by_node.append(result)
        else:
            nodes = hierarchy.get_nodes_by_level(2)
            primitives_by_node = [node.content for node in nodes]

        pipelines = []
        for primitives in primitives_by_node:
            component = random_choices(primitives, [p.weight for p in primitives])
            configuration = self.configuration_space.get_configuration_point(
                [self.learning_type], [component])
            pipelines.append(Pipeline(configuration))
        return pipelines

def pipelines_by_affinity():
    """Generate pipelines using affinity"""
    primitives = DSBoxPrimitives()
    planner = LevelOnePlanner(primitives=primitives)
    policy = AffinityPolicy(primitives)

    policy.set_symetric_affinity('Descritization', 'NaiveBayes', 10)
    policy.set_symetric_affinity('Normalization', 'SVM', 10)


    pipelines = planner.generate_pipelines_with_policy(policy, 20)
    for pipeline in pipelines:
        print(pipeline)

def pipelines_by_hierarchy(level=2):
    """Generate pipelines using tag hierarhcy"""
    primitives = D3mPrimitives()
    planner = LevelOnePlanner(primitives=primitives)

    pipelines = planner.generate_pipelines_with_hierarchy(level=level)
    for pipeline in pipelines:
        print(pipeline)

def testd3():
    """Test method"""
    primitives = D3mPrimitives()
    planner = LevelOnePlanner(primitives=primitives)

    pipelines = planner.generate_pipelines(20)
    for pipeline in pipelines:
        print(pipeline)

def print_stat():
    """Print statistics of the primitives"""
    profile = load_primitive_profile()
    classifier = [p['Name'] for p in profile
                  if 'LearningType' in p and p['LearningType']=='Classification']
    regressor = [p['Name'] for p in profile
                 if 'LearningType' in p and p['LearningType']=='Regression']

    #primitives = D3mPrimitives(classifer, regressor)
    primitives = D3mPrimitives()
    hierarchies = primitives.get_hierarchies()

    for name in Category:
        hierarchies[name].print()
    print()
    primitives.print_statistics()

def load_primitive_profile():
    """Load primitive profile"""
    # filename = pkgutil.get_data(__name__ , 'goodJSON.json')
    # with open(filename, 'r') as f:
    #     result = json.load(f)
    result = json.loads(pkgutil.get_data(__name__ , 'goodJSON.json').decode())
    for profile in result:
        if not profile['Name'] == profile['id'].split('.')[-1]:
            print (profile['Name'], profile['id'])

    return result

def compute_difference():
    good = load_primitive_profile()
    # learningType = [p for p in good if 'LearningType' in p]
    classification = set([p['Name'] for p in good
                          if 'LearningType' in p and p['LearningType'] == 'Classification'])
    regression = set([p['Name'] for p in good
                      if 'LearningType' in p and p['LearningType'] == 'Regression'])

    primitives = D3mPrimitives()
    hierarchies = primitives.get_hierarchies()
    cp = hierarchies[Category.CLASSIFICATION].get_primitives()
    classification2 = set([p.name for l in cp for p in l])
    rp = hierarchies[Category.REGRESSION].get_primitives()
    regression2 = set([p.name for l in rp for p in l])

    print("Classification: In Daniel's but not in Ke-Thia's")
    pprint.pprint(classification.difference(classification2))

    print("Classification: In Ke-Thia's but not in Daniel's")
    pprint.pprint(classification2.difference(classification))

    print("Regression: In Daniel's but not in Ke-Thia's")
    pprint.pprint(regression.difference(regression2))

    print("Regression: In Ke-Thia's but not in Daniel's")
    pprint.pprint(regression2.difference(regression))


if __name__ == "__main__":
    print_stat()
    # pipelines_by_affinity()
    pipelines_by_hierarchy()
