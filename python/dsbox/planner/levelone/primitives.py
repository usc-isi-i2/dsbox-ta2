"""Module for defining primitives and primitve categories
"""

from collections import defaultdict
from enum import Enum
import json
import pprint
import pkgutil

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
        super(DSBoxPrimitive, self).__init__()
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
        super(D3mPrimitive, self).__init__()
        self.name = definition['id'].split('.')[-1]

        self.task = ''
        if 'task_type' in definition:
            if 'feature extraction' in definition['task_type']:
                self.task = "FeatureExtraction"
            if 'data preprocessing' in definition['task_type']:
                self.task = "DataPreprocessing"

        self.learning_type = 'NA'
        if 'handles_classification' in definition and definition['handles_classification']:
            self.learning_type = 'classification'
            self.task = 'Modeling'
        if 'handles_regression' in definition and definition['handles_regression']:
            self.learning_type = 'regression'
            self.task = 'Modeling'
        self.handles_multiclass = False
        if 'handles_multiclass' in definition:
            self.handles_multiclass = definition['handles_multiclass']
        self.handles_multilabel = False
        if 'handles_multilabel' in definition:
            self.handles_multilabel = definition['handles_multilabel']

        if 'algorithm_type' in definition:
            # !!!! For now get only the first type
            self.ml_algorithm = definition['algorithm_type'][0]
            if 'graph matching' in definition['algorithm_type']:
                self.learning_type = 'graphMatching'
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
    def __init__(self, hierarchy, name, parent, content=None):
        self.hierarchy = hierarchy
        self.name = name
        self.parent = parent
        self.children = []
        self._content = content
    def get_content(self):
        return self._content
    def add_child(self, name, content=None):
        """Add a child to the hierarchy"""
        child = HierarchyNode(self.hierarchy, name, self, content)
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
    def get_siblings(self):
        if self.parent is None:
            result = []
        else:
            result = [x for x in self.parent.children if not x==self]
        return result
    def add_primitive(self, primitive):
        self.hierarchy.add_primitive(self, primitive)
    def _add_primitive(self, primitive):
        if self._content is None:
            self._content = [primitive]
        else:
            self._content.append(primitive)
    def __str__(self):
        return 'Node({},num_child={})'.format(self.name, len(self.children))

class Hierarchy(object):
    """Generic tree of nodes"""
    def __init__(self, name):
        # name of this hierarchy
        self.name = name
        self.root = HierarchyNode(self, 'root', None)
        self._changed = False
        self._level_count = []
        self.node_by_primitive = dict()

    def add_child(self, node, name, content=None):
        """Create and add child node"""
        assert node.hierarchy == self
        node.add_child(name, content)
        if content is not None:
            for primitive in content:
                self.node_by_primitive[primitive] = node
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

    def add_primitive(self, node, primitive):
        self.node_by_primitive[primitive] = node
        node._add_primitive(primitive)

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
        counts[level] = counts[level] + 1
        for child in node.children:
            counts = self._compute_level_counts(child, level+1, counts)
        return counts

    def get_primitive_count(self):
        """Returns the number of primitives"""
        return self._get_primitive_count(self.root)

    def _get_primitive_count(self, curr_node):
        """Returns the number of primitives"""
        count = 0
        if curr_node._content is not None:
            count = count + len(curr_node._content)
        for child in curr_node.children:
            count = count + self._get_primitive_count(child)
        return count

    def get_primitives(self, curr_node=None):
        if curr_node is None:
            curr_node = self.root
        result = []
        if curr_node._content is not None:
            result.append(curr_node._content)
        for child in curr_node.children:
            result = result + self.get_primitives(child)
        return result

    def get_primitives_as_list(self, curr_node=None):
        return [p for plist in self.get_primitives(curr_node) for p in plist]

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
                result = result + self._get_nodes_by_level(node, curr_level + 1, target_level)
            return result

    def get_node_by_primitive(self, primitive):
        return self.node_by_primitive[primitive]

    def pprint(self):
        """Print hierarchy"""
        print('Hierarchy({}, level_counts={})'.format(self.name, self.get_level_counts()))
        self._print(self.root, [])

    def _print(self, curr_node, path, max_depth=2):
        new_path = path + [curr_node]
        if len(new_path) > max_depth:
            print(' '*4 + ':'.join([node.name for node in new_path[1:]]))
            for line in pprint.pformat(curr_node._content).splitlines():
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
    METRICS = 7
    GRAPH = 8
    OTHER = 9

class Primitives(object):
    """Base Primitives class"""
    def __init__(self):
        self.primitives = []
        self._index = dict()
        self.size = 0
        self.hierarchy_types = [Category.PREPROCESSING, Category.FEATURE,
                                Category.CLASSIFICATION, Category.REGRESSION,
                                Category.UNSUPERVISED, Category.EVALUATION,
                                Category.METRICS, Category.GRAPH, Category.OTHER]
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
                classification = classification + 1
                # classification_algo[primitive.ml_algorithm] += 1
                classification_algo[tag_str] = classification_algo[tag_str] + 1
                tag_primitive['C:' + tag_str].append(primitive.name)
            elif primitive.learning_type == Category.REGRESSION:
                regression = regression + 1
                regression_algo[tag_str] = regression_algo[tag_str] + 1
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

    def _compute_tag_hierarchy(self):
        """Compute hierarchy based on sklearn tags"""
        for primitive in self.primitives:
            # Put base/mixin and functions into other category
            if primitive.tags[0] in ['metrics']:
                node = self.hierarchies[Category.METRICS].add_path(primitive.tags[:2])
            elif (primitive.tags[0] == 'base'
                or (primitive.tags[1] == 'base' and not 'LinearRegression' in primitive.name)
                or 'Base' in primitive.name
                or 'Mixin' in primitive.name
                or primitive.name[0].islower()
                or primitive.name == 'ForestRegressor'
                or primitive.name == 'ForestClassifier'):

                node = self.hierarchies[Category.OTHER].add_path(primitive.tags[:2])

            elif (primitive.learning_type == 'classification'
                  or primitive.tags[0] in ['lda', 'qda', 'naive_bayes']
                  or ('Classifier' in primitive.name
                      and not primitive.tags[0] in
                      ['multiclass', 'multioutput', 'calibration'])
                  or 'SVC' in primitive.name
                  or 'LogisticRegression' in primitive.name
                  or 'Perceptron' in primitive.name ):  # Same as SGDClassifier

                node = self.hierarchies[Category.CLASSIFICATION].add_path(primitive.tags[:2])

                # Modify primitive learning type
                primitive.learning_type = 'classification'

            elif (primitive.learning_type == 'regression'
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
                primitive.learning_type = 'regression'

            elif primitive.tags[0] in ['cluster', 'mixture']:

                node = self.hierarchies[Category.UNSUPERVISED].add_path(primitive.tags[:2])

            elif (primitive.tags[0] in ['feature_extraction', 'feature_selection', 'decomposition',
                                       'random_projection', 'manifold']
                  or 'OrthogonalMatchingPursuit' in primitive.name):

                node = self.hierarchies[Category.FEATURE].add_path(primitive.tags[:2])

            elif primitive.tags[0] == 'preprocessing':

                node = self.hierarchies[Category.PREPROCESSING].add_path(primitive.tags[:2])

            elif (primitive.tags[0] in ['cross_validation', 'model_selection']):

                node = self.hierarchies[Category.EVALUATION].add_path(primitive.tags[:2])

            elif (primitive.tags[0] in ['cross_validation', 'graph_matching']):

                node = self.hierarchies[Category.GRAPH].add_path(primitive.tags[:2])
            else:
                node = self.hierarchies[Category.OTHER].add_path(primitive.tags[:2])

            node.add_primitive(primitive)

class DSBoxPrimitives(Primitives):
    """Maintain available primitives"""
    PRIMITIVE_FILE = 'primitives.json'

    def __init__(self):
        super(DSBoxPrimitives, self).__init__()
        self._load()
        for index, primitive in enumerate(self.primitives):
            self._index[primitive] = index
        self.size = len(self.primitives)
        self._compute_tag_hierarchy()

    def _load(self):
        """Load primitive definition from JSON file"""
        text = pkgutil.get_data('dsbox.planner.levelone', self.PRIMITIVE_FILE)
        content = json.loads(text.decode())
        self.primitives = [DSBoxPrimitive(primitive_dict)
                           for primitive_dict in content['Primitives']]

    def _compute_tag_hierarchy(self):
        """Compute hierarchy based on sklearn tags"""
        for primitive in self.primitives:
            if primitive.learning_type == 'Classification':
                node = self.hierarchies[Category.CLASSIFICATION].add_path(
                    [primitive.ml_algorithm, primitive.ml_algorithm])
                primitive.learning_type = 'classification'

            elif primitive.learning_type == 'Regression':
                node = self.hierarchies[Category.REGRESSION].add_path(
                    [primitive.ml_algorithm, primitive.ml_algorithm])
                primitive.learning_type = 'regression'

            elif primitive.learning_type == 'UnsupervisedLearning':
                node = self.hierarchies[Category.UNSUPERVISED].add_path(
                    [primitive.ml_algorithm, primitive.ml_algorithm])

            elif primitive.task == 'FeatureExtraction':

                node = self.hierarchies[Category.FEATURE].add_path(
                    [primitive.ml_algorithm, primitive.ml_algorithm])

            elif primitive.task == 'DataPreprocessing':

                node = self.hierarchies[Category.PREPROCESSING].add_path(
                    [primitive.ml_algorithm, primitive.ml_algorithm])

            elif primitive.task == 'Evaluation':

                node = self.hierarchies[Category.EVALUATION].add_path(
                    [primitive.ml_algorithm, primitive.ml_algorithm])

            else:
                node = self.hierarchies[Category.OTHER].add_path(
                    [primitive.ml_algorithm, primitive.ml_algorithm])

            node.add_primitive(primitive)


class D3mPrimitives(Primitives):
    """Primitives from D3M metadata"""

    PRIMITIVE_FILE = 'sklearn.json'
    def __init__(self, additional_primitives):
        # additional_primitives is list of Primitives
        super(D3mPrimitives, self).__init__()
        self._load()
        if additional_primitives:
            self.primitives = self.primitives + additional_primitives
        for index, primitive in enumerate(self.primitives):
            self._index[primitive] = index
        self.size = len(self.primitives)
        self._compute_tag_hierarchy()

    def _load(self):
        """Load primitve from json"""
        text = pkgutil.get_data('dsbox.planner.levelone', self.PRIMITIVE_FILE)
        content = json.loads(text.decode())
        self.primitives = [D3mPrimitive(primitive_dict)
                           for primitive_dict in content['search_primitives']]
