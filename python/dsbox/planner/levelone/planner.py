"""This module implements the level one planner of the DSBox system.
"""
#### !!!!
import sys
import operator
sys.path.append('/home/ktyao/dev/dsbox/dsbox-ta2/python')

from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.schema.dataset_schema import VariableFileType
from dsbox.planner.levelone.primitives import Primitives, Category, DSBoxPrimitives, D3mPrimitives

import json
import pkgutil
import random
import pprint
import bisect
import itertools

import numpy as np

def random_choices(population, weights):
    """Randomly select a element based on weights. Similar to random.choices in Python 3.6+"""
    assert len(population) == len(weights)
    total_weight = np.sum(weights)
    rand = total_weight * random.random()
    i = 0
    while rand > weights[i]:
        rand -= weights[i]
        i += 1
    return population[i]

def random_choices_without_replacement(population, weights, k=1):
    """Randomly sample multiple element based on weights witout replacement."""
    assert len(weights) == len(population)
    if k > len(population):
        k = len(population)
    weights = list(weights)
    result = []
    for index in range(k):
        cum_weights = list(accumulate(weights))
        total = cum_weights[-1]
        i = bisect.bisect(cum_weights, random.random() * total)
        result.append(population[i])
        weights[i] = 0
    return result

def accumulate(iterable, func=operator.add):
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total

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
        for component_space in self.space:
            if components:
                values = policy.get_affinities(components, component_space)
                component = random_choices(component_space, values)
            else:
                i = random.randrange(len(component_space))
                component = component_space[i]
            components.append(component)
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
    def get_dim_value(self, name):
        if not name in self.configuration_space.dimension_names:
            raise Exception('{} is not a configuration space dimension name'.format(name))
        index = self.configuration_space.dimension_names.index(name)
        return self.point[index]
    def new_point_replace(self, name, value):
        """Generate new point by replacing coordiante at dimension name with new value"""
        new_point = list(self.point)
        if not name in self.configuration_space.dimension_names:
            raise Exception('{} is not a configuration space dimension name'.format(name))
        index = self.configuration_space.dimension_names.index(name)
        new_point[index] = value
        return ConfigurationPoint(self.configuration_space, new_point)

class AffinityPolicy(object):
    """Defines affinity pairs of components.
    The default affinity is 1. If affinity value is 10 then the pair
    is 10x more likely to occur together compared to default.
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

    def get_affinities(self, source_primitives, dest_primitives,
                       source_weights=None, dest_weights=None):
        """Returns vector of affinitites of length len(dest_primitives)"""
        source_index = [self.primitives.get_index(p) for p in source_primitives]
        dest_index = [self.primitives.get_index(p) for p in dest_primitives]
        if source_weights is None:
            source_weights = [p.weight for p in source_primitives]
        if dest_weights is None:
            dest_weights = [p.weight for p in dest_primitives]
        result = np.empty(len(dest_primitives))
        for pos, (dst, d_weight)  in enumerate(zip(dest_index, dest_weights)):
            affinity_sum = 1e-8
            for src, s_weight in zip(source_index, source_weights):
                affinity_sum += self.affinity_matrix[src, dst] * s_weight * d_weight
            result[pos] = affinity_sum
        return result

class Pipeline(object):
    """Defines a sequence of executions"""
    def __init__(self, configuration_point):
        self.configuration_point = configuration_point

    def get_primitive(self, dim_name):
        return self.configuration_point.get_dim_value(dim_name)

    def get_primitives(self):
        return [primitive for primitive in self.configuration_point.point
                if primitive is not None]

    def new_pipeline_replace(self, dim_name, new_component):
        """Generate new pipeline by replacing primitive at dimension name with new primitive"""
        point = self.configuration_point.new_point_replace(dim_name, new_component)
        return Pipeline(point)

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
    def __init__(self, task_type=TaskType.CLASSIFICATION,
                 task_subtype=TaskSubType.BINARY,
                 metric=Metric.F1,
                 media_type = VariableFileType.NONE,
                 ignore_preprocessing=True,
                 primitives=None,
                 policy=None):
        self.task_type = task_type
        self.task_subtype = task_subtype
        self.media_type = media_type
        self.evaluator_name = metric
        self.ignore_preprocessing = ignore_preprocessing
        if primitives:
            self.primitives = primitives
        else:
            self.primitives = Primitives()
        self.configuration_space = self.compute_configuration_space()
        self.policy = policy

    def compute_configuration_space(self):
        """Compute configuration space using Primitives"""
        dimension_name = []
        dimension = []
        if not self.ignore_preprocessing:
            dimension_name.append('DataPreprocessing')
            preprocess = self.primitives.hierarchies[Category.PREPROCESSING].get_primitives_as_list()
            dimension.append(preprocess)

        dimension_name.append('FeatureExtraction')
        feature = self.primitives.hierarchies[Category.FEATURE].get_primitives_as_list()
        dimension.append(feature)

        dimension_name.append(self.task_type.value)
        if self.task_type == TaskType.CLASSIFICATION:
            learner = self.primitives.hierarchies[Category.CLASSIFICATION].get_primitives_as_list()
        else:
            learner = self.primitives.hierarchies[Category.REGRESSION].get_primitives_as_list()
        dimension.append(learner)

        dimension_name.append('Metrics')
        evaluator = self.primitives.hierarchies[Category.METRICS].get_primitives_as_list()
        dimension.append(evaluator)

        return ConfigurationSpace(dimension_name, dimension)

    # def compute_configuration_space(self):
    #     """Compute configuration space using Primitives"""
    #     dimension_name = []
    #     dimension = []
    #     if not self.ignore_preprocessing:
    #         dimension_name.append('DataPreprocessing')
    #         preprocess = self.primitives.filter_by_task('DataPreprocessing')
    #         dimension.append(preprocess)

    #     dimension_name.append('FeatureExtraction')
    #     feature = self.primitives.filter_by_task('FeatureExtraction')
    #     dimension.append(feature)

    #     dimension_name.append(self.task_type.value)
    #     learner = self.primitives.filter_by_learning_type(self.task_type.value)
    #     dimension.append(learner)

    #     dimension_name.append('Evaluation')
    #     evaluator = [self.primitives.get_by_name(self.evaluator_name.value)]
    #     dimension.append(evaluator)

    #     return ConfigurationSpace(dimension_name, dimension)

    def get_primitive_weight(self, primitive, hierarchy):
        if not hierarchy.name == Category.FEATURE:
            return primitive.weight
        if not (self.media_type == VariableFileType.IMAGE
            or self.media_type == VariableFileType.TEXT
            or self.media_type == VariableFileType.AUDIO):
            return primitive.weight

        factor = 100
        node = hierarchy.get_node_by_primitive(primitive)
        if ((self.media_type == VariableFileType.IMAGE and node.name == 'image')
            or (self.media_type == VariableFileType.TEXT and node.name == 'text')
            or (self.media_type == VariableFileType.AUDIO and node.name == 'audio')):
            return factor * primitive.weight
        else:
            return primitive.weight

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

    def find_primitives_by_hierarchy(self, dim_name, hierarchy, level=2):
        """Returns one random primitive per node in the hierarchy"""
        if level == 1:
            nodes = hierarchy.get_nodes_by_level(1)
            primitives_by_node = []
            for l1_node in nodes:
                result = []
                for l2_node in l1_node.children:
                    result += l2_node.get_content()
                primitives_by_node.append(result)
        else:
            nodes = hierarchy.get_nodes_by_level(2)
            primitives_by_node = [node.get_content() for node in nodes]
        result = []
        for primitives in primitives_by_node:
            weights = [self.get_primitive_weight(p, hierarchy) for p in primitives]
            component = random_choices(primitives, weights)
            result.append(component)
        return result

    def generate_pipelines_with_hierarchy(self, level=2):
        """Generate singleton pipeline using tag hierarchy"""
        if self.task_type == TaskType.CLASSIFICATION:
            learning_type = TaskType.CLASSIFICATION.value  # 'classification'
            hierarchy = self.primitives.hierarchies[Category.CLASSIFICATION]
        elif self.task_type == TaskType.REGRESSION:
            learning_type = TaskType.REGRESSION.value  # 'regression'
            hierarchy = self.primitives.hierarchies[Category.REGRESSION]
        else:
            raise Exception('Learning type not recoginized: {}'.format(self.task_type))

        primitives = self.find_primitives_by_hierarchy(
            learning_type, hierarchy, level)
        pipelines = []
        for component in primitives:
            configuration = self.configuration_space.get_configuration_point(
                [learning_type], [component])
            pipelines.append(Pipeline(configuration))
        return pipelines

    def fill_feature_with_hierarchy(self, pipeline, level=2):
        """Return new pipelines by filling in the feature extraction component
        using the primitive hierarchy"""
        dimension_name = 'FeatureExtraction'
        current_primitive = pipeline.get_primitive(dimension_name)
        pipeline_primitives = [primitive for primitive in pipeline.get_primitives()
                               if not primitive==current_primitive]
        hierarchy = self.primitives.hierarchies[Category.FEATURE]
        primitives = self.find_primitives_by_hierarchy(dimension_name, hierarchy, level)

        new_pipelines = []
        for feature_component in primitives:
            new_pipelines.append(pipeline.new_pipeline_replace(dimension_name, feature_component))
        return new_pipelines

    def fill_feature_by_weights(self, pipeline, num_pipelines=5):
        dimension_name = 'FeatureExtraction'
        current_primitive = pipeline.get_primitive(dimension_name)
        pipeline_primitives = [primitive for primitive in pipeline.get_primitives()
                               if not primitive==current_primitive]
        hierarchy = self.primitives.hierarchies[Category.FEATURE]
        all_primitives = hierarchy.get_primitives_as_list()
        all_primitives = [primitive for primitive in all_primitives if not primitive == current_primitive]
        primitive_weights = [self.get_primitive_weight(primitive, hierarchy)
                             for primitive in all_primitives]
        primitives = random_choices_without_replacement(all_primitives, primitive_weights, num_pipelines)
        new_pipelines = []
        for feature_component in primitives:
            new_pipelines.append(pipeline.new_pipeline_replace(dimension_name, feature_component))
        return new_pipelines


    def find_similar_learner(self, pipeline, include_siblings=False, num_pipelines=5):
        """Fine similar pipelines by replacing the learner component"""
        if self.task_type == TaskType.CLASSIFICATION:
            learning_type = TaskType.CLASSIFICATION.value  # 'classification'
            hierarchy = self.primitives.hierarchies[Category.CLASSIFICATION]
        elif self.task_type == TaskType.REGRESSION:
            learning_type = TaskType.REGRESSION.value  # 'regression'
            hierarchy = self.primitives.hierarchies[Category.REGRESSION]
        else:
            raise Exception('Learning type not recoginized: {}'.format(self.task_type))
        learner = pipeline.get_primitive(learning_type)
        learner_node = hierarchy.get_node_by_primitive(learner)
        if include_siblings:
            nodes = [learner_node] + learner_node.get_siblings()
        else:
            nodes = [learner_node]
        similar_primitives = [primitive for node in nodes for primitive in node.get_content()]
        pipeline_primitives = [primitive for primitive in pipeline.get_primitives() if not primitive==learner]
        similar_weights = [self.get_primitive_weight(primitive, hierarchy)
                           for primitive in similar_primitives]
        pipeline_weights = [self.get_primitive_weight(primitive, hierarchy)
                            for primitive in pipeline_primitives]
        values = self.policy.get_affinities(pipeline_primitives, similar_primitives,
                                            pipeline_weights, similar_weights)
        similar_primitives = random_choices_without_replacement(similar_primitives, values, num_pipelines)

        similiar_pipelines = []
        for component in similar_primitives:
            similiar_pipelines.append(pipeline.new_pipeline_replace(learning_type, component))
        return similiar_pipelines

def get_d3m_primitives():
    #from dsbox.primitive import DSBOX_PRIMITIVES
    additional_primitives = []
    #for pclass in DSBOX_PRIMITIVES:
    #    additional_primitives.append(pclass().annotation())
    ps = D3mPrimitives(additional_primitives)
    return ps

def pipelines_by_affinity():
    """Generate pipelines using affinity"""
    primitives = DSBoxPrimitives()

    policy = AffinityPolicy(primitives)
    policy.set_symetric_affinity(primitives.get_by_name('Descritization'),
                                 primitives.get_by_name('NaiveBayes'), 10)
    policy.set_symetric_affinity(primitives.get_by_name('Normalization'),
                                 primitives.get_by_name('SVM'), 10)

    planner = LevelOnePlanner(primitives=primitives, policy=policy)

    pipelines = planner.generate_pipelines_with_policy(policy, 20)
    for pipeline in pipelines:
        print(pipeline)

def pipelines_by_hierarchy(level=2):
    """Generate pipelines using tag hierarhcy"""
    primitives = get_d3m_primitives()
    policy = AffinityPolicy(primitives)
    planner = LevelOnePlanner(primitives=primitives, policy=policy)

    pipelines = planner.generate_pipelines_with_hierarchy(level=level)
    for pipeline in pipelines:
        print(pipeline)
    return pipelines

def testd3():
    """Test method"""
    primitives = get_d3m_primitives()
    planner = LevelOnePlanner(primitives=primitives)

    pipelines = planner.generate_pipelines(20)
    for pipeline in pipelines:
        print(pipeline)

def print_stat():
    """Print statistics of the primitives"""
    # profile = load_primitive_profile()
    # classifier = [p['Name'] for p in profile
    #               if 'LearningType' in p and p['LearningType']=='Classification']
    # regressor = [p['Name'] for p in profile
    #              if 'LearningType' in p and p['LearningType']=='Regression']

    # primitives = D3mPrimitives(classifer, regressor)
    primitives = get_d3m_primitives()  # D3mPrimitives()
    hierarchies = primitives.get_hierarchies()

    for name in Category:
        hierarchies[name].pprint()
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

    primitives = get_d3m_primitives()  # D3mPrimitives()
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


#if __name__ == "__main__":
#    print_stat()
    # pipelines_by_affinity()
#    pipelines_by_hierarchy()
