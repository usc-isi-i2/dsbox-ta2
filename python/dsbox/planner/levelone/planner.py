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

class Primitives(object):
    """Maintain available primitives"""
    PRIMITIVE_FILE = 'primitives.json'

    def __init__(self):
        self._load()
        self._index = dict()
        for index, primitive in enumerate(self.primitives):
            self._index[primitive['Name']] = index
        self.size = len(self.primitives)

    def _load(self):
        """Load primitive definition from JSON file"""
        text = pkgutil.get_data('dsbox.planner.levelone', self.PRIMITIVE_FILE)
        content = json.loads(text.decode())
        self.primitives = content['Primitives']

    def filter_equality(self, aspect, name):
        """Find primitive by aspect and name value"""
        result = [p for p in self.primitives if p[aspect] == name]
        return result

    def filter_by_task(self, name):
        """Find primitive by task aspect and name value"""
        return self.filter_equality('Task', name)

    def filter_by_learning_type(self, name):
        """Find primitive by learning-type aspect and name value"""
        return self.filter_equality('LearningType', name)

    def filter_by_algo(self, name):
        """Find primitive by algorithm aspect and name value"""
        return self.filter_equality('MachineLearningAlgorithm', name)

    def get_by_name(self, name):
        """Get primitve by unique name"""
        for primitive in self.primitives:
            if primitive['Name'] == name:
                return primitive
        return None

    def get_index(self, name):
        """Returns the index of the primitive given its name"""
        return self._index[name]

class ConfigurationSpace(object):
    """Defines the space of pipelines"""

    def __init__(self, dimension_names, space):
        assert len(dimension_names) == len(space)
        self.dimension_names = dimension_names
        self.space = space

    def get_random_configuration(self, seed=None):
        """Returns a random configuration from the configuration space"""
        if seed:
            random.seed(seed)
        components = []
        for component_space in self.space:
            i = random.randrange(len(component_space))
            component = component_space[i]
            components.append(component)
        return components

    def get_configuration_by_policy(self, policy):
        """Returns a random configuration based on stochastic policy"""
        components = []
        component_names = []
        for component_space in self.space:
            names = [comp['Name'] for comp in component_space]
            if components:
                values = np.power(10,
                                  policy.get_affinities(component_names, names))
                rand = random.randrange(np.sum(values))
                i = 0
                while rand > values[i]:
                    rand -= values[i]
                    i += 1
            else:
                i = random.randrange(len(component_space))
            component = component_space[i]
            components.append(component)
            component_names.append(component['Name'])
        return components


class AffinityPolicy(object):
    """Defines affinity pairs of components"""
    def __init__(self, primitives):
        self.primitives = primitives
        self.affinity_matrix = np.zeros((primitives.size, primitives.size))

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
    def __init__(self, component_list):
        self.component_list = component_list

    @classmethod
    def get_random_pipeline(cls, configuration_space):
        """Returns a random pipeline"""
        return Pipeline(configuration_space.get_random_configuration())

    def __str__(self):
        return 'Pipeline(' + ','.join([c['Name'] for c in self.component_list]) + ')'

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

    policy.set_symetric_affinity('Descritization', 'NaiveBayes', 1)
    policy.set_symetric_affinity('Normalization', 'SVM', 1)


    pipelines = planner.generate_pipelines_with_policy(policy, 20,
                                                       ignore_preprocessing=False)
    for pipeline in pipelines:
        print(pipeline)

if __name__ == "__main__":
    main()
