"""This module implements the level one planner of the DSBox system.
"""

from dsbox.planner.common import D3MOntology, D3MPrimitiveLibrary
from dsbox.planner.common.pipeline import Pipeline
from dsbox.planner.common.primitive import Primitive
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.schema.dataset_schema import VariableFileType
from dsbox.planner.levelone.primitives import Primitives, Category, DSBoxPrimitives, D3mPrimitives

from d3m_metadata.metadata import PrimitiveFamily

import bisect
import json
import numpy as np
import operator
import os
import pkgutil
import pprint
import random
import typing

import pdb

from warnings import warn

class LevelOnePlanner(object):
    '''Level One Planner'''
    def __init__(self, * ,
                 primitive_library : D3MPrimitiveLibrary,
                 ontology : D3MOntology,
                 library_dir : str,
                 task_type : TaskType,
                 task_subtype : TaskSubType,
                 media_type : VariableFileType = VariableFileType.NONE,
                 include = [],
                 exclude = []):
        self.primitive_library = primitive_library
        self.ontology = ontology
        self.library_dir = library_dir
        self.task_type = task_type
        self.media_type = media_type
        self.primitive_family_mappings = PrimitiveFamilyMappings()
        self.primitive_family_mappings.load_json(library_dir)

        if not include and not exclude:    
            self.include = False
        else:
            self.include = True
            self.include_families, self.include_types, self.include_primitives = self._interpret_inc_exc(include)
            self.exclude_families, self.exclude_types, self.exclude_primitives = self._interpret_inc_exc(exclude, inc = False)

    def _interpret_inc_exc(self, mixed_list, inc = True):
        # mixed_list can include families and/or primitives
        # we would like to exclude families or single primitives within families
        # if including a primitive, we need its family to understand where in pipeline it is used 
            # e.g. choose a featurizer and keep classification search
        family_list = []
        primitive_list = []
        type_list = []
        for entry in mixed_list:
            if entry in list(self.primitive_library.primitives_by_family.keys()):
                family_list.append(entry)
                
            elif entry in list(self.primitive_library.primitive_by_package.keys()):
                primitive_list.append(entry)
            
            elif entry in list(self.primitive_library.primitives_by_type.keys()):
                type_list.append(str(entry))
            
            else:
                print('Could not find include/exclude item: ', entry)

        return family_list, type_list, primitive_list


    def generate_pipelines_with_hierarchy(self, level=2) -> typing.List[Pipeline]:
        # ignore level for now, since only have one level hierarchy
        results = []
        families = self.primitive_family_mappings.get_families_by_task_type(self.task_type.value)
        if self.task_type == TaskType.GRAPH_MATCHING:
            # if GRAPH_MATCHING then add LINK_PREDICTION
            families = families + self.primitive_family_mappings.get_families_by_task_type(TaskType.LINK_PREDICTION)

        # moved to get_child_nodes
        #families = sebt(families)-set(self.exclude_families   
        family_nodes = [self.ontology.get_family(f) for f in families]
        
        # include / exclude family and type checks
        child_nodes = self._get_child_nodes(family_nodes)

        for node in child_nodes:
            primitives = self.ontology.hierarchy.get_primitives_as_list(node)

            if not primitives:
                continue
                
            # Inclusion checks
            
            primitives = [p for p in primitives if self._check_primitive_include(p)]
            
            if not primitives:
                continue

            weights = [p.weight for p in primitives]
        
            # Set task type for execute_pipeline
            for p in primitives:
                p.task = 'Modeling'
        
            primitive = random_choices(primitives, weights)
            pipe = Pipeline(primitives=[primitive])
            results.append(pipe)
        return results

    def _get_child_nodes(self, family_nodes):
        child_nodes = []
        family_types = []
        
        try:
            family_nodes = [f for f in family_nodes if f.name not in self.exclude_families] 
        except:
            pass

        incl_prim_types = []
        
        if hasattr(self, 'include_primitives'):
            for p in self.include_primitives:
                #need the prim 
                try:
                    p = self.ontology.hierarchy.node_by_primitive[p]._content[0]
                except:
                    continue
                incl_prim_types.extend(list(p.getAlgorithmTypes()))
        
        # check family / type inclusions / exclusions here
        for node in family_nodes:
            # include primitives will get through here because family included for each
            if not self.include:
                child_nodes += node.get_children()
            elif node.name in self.include_families:
                child_nodes += [n for n in node.get_children() if n.name not in self.exclude_types]
            elif node.name not in self.exclude_families:                
                for child in node.get_children():
                    family_types.append(child.name)
                    # add child if it is in include_types
                    if child.name in self.include_types or child.name in incl_prim_types:
                        print(child.name)
                        child_nodes.append(child)

                # if types regard different task, then continue (e.g. featurization types, classification family)
                if self.include_types and set(family_types).isdisjoint(self.include_types):
                    child_nodes += node.get_children()
                #if self.include_primitives and set(family_types).isdisjoint(set( ))
                if not set(family_types).isdisjoint(self.exclude_types):
                    print("Excluding types: ", self.exclude_types, " incl: ", set(node.get_children()) - set(self.exclude_types))
                    child_nodes += (set(node.get_children()) - set(self.exclude_types))

        return child_nodes

    def _check_primitive_include(self, p):
        if not self.include:
            return True

        check1 = (self.include_families or self.include_types or self.include_primitives)
        check2 = p.getFamily() in self.include_families
        check3 = self.include_types and not set(p.getAlgorithmTypes()).isdisjoint(self.include_types) 
        check4 = p.cls in self.include_primitives
        
        check = check1 and (check2 or check3 or check4)

        if p.cls in self.exclude_primitives:
            return False
        else:
            return check


    def fill_feature_by_weights(self, pipeline : Pipeline, num_pipelines=5) -> Pipeline:
        """Insert feature primitive weighted by on media_type"""
        selected_primitives = []
        
        feature_primitive_paths = self.primitive_family_mappings.get_primitives_by_media(
            self.media_type)
        
        feature_primitives = []
        for path in feature_primitive_paths:
            if self.primitive_library.has_primitive_by_package(path):
                feature_primitives.append(self.primitive_library.get_primitive_by_package(path))
            else:
                new_primitive = self.primitive_library.add_custom_primitive(path)
                if new_primitive:
                    print('Adding custom primitive to library: {}'.format(path))
                    feature_primitives.append(new_primitive)
                else:
                    print('Library does not have primitive {}'.format(path))
                    print('Possible error in file primitive_family_mappings.json')
        primitive_weights = [p.weight for p in feature_primitives]
        
        if feature_primitives:
            selected_primitives.extend(random_choices_without_replacement
                (feature_primitives, primitive_weights, num_pipelines))
        
        new_pipelines = []
        for p in selected_primitives:
            # Set task type for execute_pipeline
            p.task = 'FeatureExtraction'
            
            newp = pipeline.clone()
            newp.insertPrimitiveAt(0, p)
            new_pipelines.append(newp)
            
        if new_pipelines:
            return new_pipelines
        else:
            return [pipeline.clone()]

    ## Empty families for all media types. Does not work
    # def fill_feature_by_weights(self, pipeline : Pipeline, num_pipelines=5) -> Pipeline:
    #     """Insert feature primitive weighted by on media_type"""
    #     feature_primitives = [
    #         primitive
    #         for family in self.primitive_family_mappings.get_families_by_media(self.media.value)
    #         for primitive in self.primitive_library.get_primitives_by_family(family)]

    #     primitive_weights = self._get_feature_weights(feature_primitives)
    #     selected_primitives = random_choices_without_replacement(
    #         feature_primitives, primitive_weights, num_pipelines)
    #     new_pipelines = [pipeline.clone().insertPrimitiveAt(0, p) for p in selected_primitives]
    #     return new_pipelines

    def find_similar_learner(self, pipeline : Pipeline, include_siblings=True,
                            num_pipelines=5, position=-1) -> typing.List[Pipeline]:
        '''Use ontology to find similar learners'''
        # Assume learner is last primitive in pipeline
        if position < 0:
            position = pipeline.length() - 1
        learner = pipeline.getPrimitiveAt(position)

        # primitives under the same subtree are similar
        nodes = [self.ontology.hierarchy.get_node_by_primitive(learner)]

        # primitives under sibiling subtrees are similar
        if include_siblings:
            nodes = nodes + nodes[0].get_siblings()

        # Get primitives but remove the learner itself
        primitives_list = [self.ontology.hierarchy.get_primitives_as_list(n) for n in nodes]
        if learner in primitives_list[0]:
            primitives_list[0].remove(learner)

        # Weight primitives under the same subtree more
        factor = 10
        weights = [factor*p.weight for p in primitives_list[0]]
        for primitives in primitives_list[1:]:
            weights = weights + [p.weight for p in primitives]
        selected = random_choices_without_replacement(
            [p for primitives in primitives_list for p in primitives],
            weights, num_pipelines)
        new_pipelines = []
        for p in selected:
            # Set task type for execute_pipeline
            p.task = 'Modeling'
            
            if not self._check_primitive_include(p):
                continue

            new_pipeline = pipeline.clone()
            new_pipeline.replacePrimitiveAt(position, p)
            new_pipelines.append(new_pipeline)
        return new_pipelines
	
    def extend_pipeline_with_feature_selection(self, l2_pipeline) -> typing.List[Pipeline]:
        """
        get a feature selection primitive;
        insert it to the last second position of l2_pipeline (right before the estimator)
        """

        result = []
        result.append(l2_pipeline)
        l2_pipeline_copy = l2_pipeline.clone()
        
        family_node = self.ontology.get_family("FEATURE_SELECTION")
        child_nodes = family_node.get_children() # different methods of feature selection

        for node in child_nodes:
            primitives = self.ontology.hierarchy.get_primitives_as_list(node)
            if not primitives:
                continue
            # after getting the primitive
            p = primitives[0] # fixed, just use the first one
            l2_pipeline_copy.insertPrimitiveAt(-1, p)
            result.append(l2_pipeline_copy)
            return result

    def _get_feature_weights(self, primitives : typing.List[Primitive]):
        factor = 50
        weights = []
        media_families = self.primitive_family_mappings.media_to_family[self.media_type.value]
        for primitive in primitives:
            family = primitive.getFamily()
            if family in media_families:
                weights.append(factor * primitive.weight)
        return weights

class PrimitiveFamilyMappings(object):
    '''Mappings to find primitive families related to specific task, and to specific media'''
    def __init__(self):
        self.task_to_family : typing.Dict[str, str]= dict()
        self.media_to_family : typing.Dict[str, str] = dict()
        self.media_to_primitives : typing.Dict[str, str] = dict()

    def load_json(self, library_dir, filename='primitive_family_mappings.json'):
        path = os.path.join(library_dir, filename)
        with open(path) as fp:
            definition = json.load(fp)
            self.task_to_family = definition['task_to_primitive_family']
            self.media_to_family = definition['media_to_primitive_family']
            self.media_to_primitives = definition['media_to_primitives']

    def get_families_by_task_type(self, task_type : str) -> typing.List[str]:
        '''Given taskType name'''
        if isinstance(task_type, str):
            return self.task_to_family[task_type]
        else:
            # TaskType
            return self.task_to_family[task_type.value]

    def get_families_by_media(self, media : str) -> typing.List[str]:
        '''Given VariableFileType names return list of primitive family names'''
        if isinstance(media, str):
            return self.media_to_family[media] + self.media_to_family['generic']
        else:
            return self.media_to_family[media.value] + self.media_to_family['generic']

    def get_primitives_by_media(self, media : str) -> typing.List[str]:
        '''Given VariableFileType names return list of primitives'''
        if isinstance(media, str):
            return self.media_to_primitives[media]
        else:
            return self.media_to_primitives[media.value]

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
    """Primitive ontology"""
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

class PipelineOld(object):
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
        return PipelineOld(point)

    @classmethod
    def get_random_pipeline(cls, configuration_space):
        """Returns a random pipeline"""
        return PipelineOld(configuration_space.get_random_configuration())

    def __str__(self):
        out_list = []
        for name, component in zip(self.configuration_point.configuration_space.dimension_names,
                                   self.configuration_point.point):
            if component is None:
                out_list.append('{}=None'.format(name))
            else:
                out_list.append('{}={}'.format(name, component.name))
        return 'PipelineOld(' + ', '.join(out_list) + ')'

class LevelOnePlannerOld(object):
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

        learner = None
        if self.task_type == TaskType.CLASSIFICATION:
            learner = self.primitives.hierarchies[Category.CLASSIFICATION].get_primitives_as_list()
        elif self.task_type == TaskType.REGRESSION:
            learner = self.primitives.hierarchies[Category.REGRESSION].get_primitives_as_list()
        elif self.task_type == TaskType.GRAPH_MATCHING:
            learner = self.primitives.hierarchies[Category.GRAPH].get_primitives_as_list()
        elif self.task_type == TaskType.TIME_SERIES_FORECASTING:
            # FIXME: assume time series forecasting is regression
            learner = self.primitives.hierarchies[Category.REGRESSION].get_primitives_as_list()
            # FIXME: Change task_type to regression
            self.task_type = TaskType.REGRESSION
        else:
            print('L1 Planner: task type "{}" not implemented'.format(self.task_type))

        if learner is not None:
            dimension_name.append(self.task_type.value)
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
            or self.media_type == VariableFileType.AUDIO
            or self.media_type == VariableFileType.TIMESERIES):
            return primitive.weight

        factor = 100
        node = hierarchy.get_node_by_primitive(primitive)
        if ((self.media_type == VariableFileType.IMAGE and node.name == 'image')
            or (self.media_type == VariableFileType.TEXT and node.name == 'text')
            or (self.media_type == VariableFileType.AUDIO and node.name == 'audio')
            or (self.media_type == VariableFileType.TIMESERIES and node.name == 'timeseries')):
            return factor * primitive.weight
        else:
            return primitive.weight

    def generate_pipelines(self, num_pipelines=5):
        """Generation pipelines"""

        pipelines = []
        for _ in range(num_pipelines):
            pipeline = PipelineOld.get_random_pipeline(self.configuration_space)
            pipelines.append(pipeline)

        return pipelines

    def generate_pipelines_with_policy(self, policy, num_pipelines=5):
        """Generate pipelines using affinity policy"""

        pipelines = []
        for _ in range(num_pipelines):
            configuration = self.configuration_space.get_configuration_by_policy(policy)
            pipeline = PipelineOld(configuration)
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
        elif self.task_type == TaskType.GRAPH_MATCHING:
            learning_type = TaskType.GRAPH_MATCHING.value  # 'graphMatching'
            hierarchy = self.primitives.hierarchies[Category.GRAPH]
        elif self.task_type == TaskType.TIME_SERIES_FORECASTING:
            # FIXME: For now assume all time series forecasting are regression problems
            learning_type = TaskType.REGRESSION.value
            hierarchy = self.primitives.hierarchies[Category.REGRESSION]
        else:
            raise Exception('Learning type not recoginized: {}'.format(self.task_type))

        primitives = self.find_primitives_by_hierarchy(
            learning_type, hierarchy, level)
        pipelines = []
        for component in primitives:
            configuration = self.configuration_space.get_configuration_point(
                [learning_type], [component])
            pipe = PipelineOld(configuration)
            pipelines.append(pipe)
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

    planner = LevelOnePlannerOld(primitives=primitives, policy=policy)

    pipelines = planner.generate_pipelines_with_policy(policy, 20)
    for pipeline in pipelines:
        print(pipeline)

def pipelines_by_hierarchy(level=2):
    """Generate pipelines using tag hierarchy"""
    primitives = get_d3m_primitives()
    policy = AffinityPolicy(primitives)
    planner = LevelOnePlannerOld(primitives=primitives, policy=policy)

    pipelines = planner.generate_pipelines_with_hierarchy(level=level)
    for pipeline in pipelines:
        print(pipeline)
    return pipelines

def testd3():
    """Test method"""
    primitives = get_d3m_primitives()
    planner = LevelOnePlannerOld(primitives=primitives)

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
