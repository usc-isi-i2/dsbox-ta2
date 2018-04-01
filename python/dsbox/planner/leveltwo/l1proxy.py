import json
import os

from dsbox.planner.levelone.planner import (LevelOnePlanner, get_d3m_primitives, AffinityPolicy)
from dsbox.planner.common.library import PrimitiveLibrary
from dsbox.planner.common.pipeline import Pipeline
from dsbox.planner.common.ontology import D3MOntology
from dsbox.schema.dataset_schema import VariableFileType
from dsbox.schema.profile_schema import DataProfileType as dpt

from dsbox.planner.common.library import D3MPrimitiveLibrary

class LevelOnePlannerProxy(object):
    """
    The Level-1 DSBox Proxy Planner.

    This is here to integrate with Ke-Thia's L1 Planner until we come up with a consistent interface
    """
    def __init__(self, libdir, helper, include = [], exclude = []):
        # Load primitives library
        self.primitives = D3MPrimitiveLibrary()

        # First load black listed primtives
        primitive_black_list_file = os.path.join(libdir, 'black-list.json')
        self.primitives.load_black_list(primitive_black_list_file)

        # Load d3m primitives
        self.primitives.load_from_d3m_index()

        # Augment with primitive annotations from Daniel
        profile_file = os.path.join(libdir, 'profile_output.json')
        self.add_primitive_requirements(profile_file)

        # Load ontology
        self.ontology = D3MOntology(self.primitives)
        self.ontology.load_curated_hierarchy(libdir)

        #self.media_type = VariableFileType.GENERIC       
        self.media_type = VariableFileType.NONE
        if helper.data_manager.media_type is not None:
            self.media_type = helper.data_manager.media_type

        self.l1_planner = LevelOnePlanner(primitive_library=self.primitives,
                                          ontology=self.ontology,
                                          library_dir=libdir,
                                          task_type=helper.problem.task_type,
                                          task_subtype=helper.problem.task_subtype,
                                          media_type=self.media_type,
                                          include = include,
                                          exclude = exclude)

    def add_primitive_requirements(self, profile_file):
        # Augment primitive metadata with Daniel's primitive profiler output
        with open(profile_file) as fp:
            primitive_profiles = json.load(fp)

        all_preconditions = set()
        for package, profile in primitive_profiles.items():
            if not self.primitives.has_primitive_by_package(package):
                #print('Cannot find class: {}'.format(package))
                continue
            primitive = self.primitives.get_primitive_by_package(package)
            if 'Requirements' in profile:
                all_preconditions |= {x for x in profile['Requirements']}
                for x in profile['Requirements']:
                    if x == "NO_MISSING_VALUES":
                        primitive.addPrecondition({dpt.MISSING_VALUES: False})
                    elif x == "NO_CATEGORICAL_VALUES":
                        primitive.addPrecondition({dpt.NUMERICAL: True})
                    elif x == "POSITIVE_VALUES":
                        primitive.addPrecondition({dpt.NEGATIVE: False})
            if 'Error' in profile:
                primitive.addErrorCondition({x:True for x in profile['Error']})

    def get_pipelines(self, data):
        try:
            print('getting pipelines with hierarchy l1 proxy')
            l1_pipelines = self.l1_planner.generate_pipelines_with_hierarchy(level=2)
            print("finished l1proxy get pipelines /n /n")
            print(l1_pipelines)
            # If there is a media type, use featu
            new_pipes = []
            for l1_pipeline in l1_pipelines:
                refined_pipes = self.l1_planner.fill_feature_by_weights(l1_pipeline, 1)
                new_pipes = new_pipes + refined_pipes

            l1_pipelines = new_pipes
            return l1_pipelines
        except Exception as _:
            return None

    def get_related_pipelines(self, pipeline):
        pipelines = self.l1_planner.find_similar_learner(pipeline, include_siblings=True)
        return pipelines

class LevelOnePlannerProxyOld(object):
    """
    The Level-1 DSBox Proxy Planner.

    This is here to integrate with Ke-Thia's L1 Planner until we come up with a consistent interface
    """
    def __init__(self, libdir, helper):
        self.models = PrimitiveLibrary(libdir + os.sep + "models.json")
        self.features = PrimitiveLibrary(libdir + os.sep + "features.json")

        self.primitives = get_d3m_primitives()
        self.policy = AffinityPolicy(self.primitives)
        self.media_type = None
        if helper.data_manager.media_type is not None:
            self.media_type = helper.data_manager.media_type

        self.l1_planner = LevelOnePlanner(primitives=self.primitives, policy=self.policy,
                task_type=helper.problem.task_type, task_subtype=helper.problem.task_subtype, media_type=self.media_type)

        self.primitive_hash = {}
        for model in self.models.primitives:
            self.primitive_hash[model.name] = model
        for feature in self.features.primitives:
            self.primitive_hash[feature.name] = feature

        self.pipeline_hash = {}

    def get_pipelines(self, data):
        try:
            print('getting pipelines with hierarchy l1 proxy')
            l1_pipelines = self.l1_planner.generate_pipelines_with_hierarchy(level=2)
            print("finished l1proxy get pipelines /n /n")
            print(l1_pipelines)
            # If there is a media type, use featurisation-added pipes instead
            # kyao: added check to skip if media_type is nested tables
            if self.media_type and not (self.media_type==VariableFileType.TABULAR or self.media_type==VariableFileType.GRAPH):
                new_pipes = []
                for l1_pipeline in l1_pipelines:
                    refined_pipes = self.l1_planner.fill_feature_by_weights(l1_pipeline, 1)
                    new_pipes = new_pipes + refined_pipes
                l1_pipelines = new_pipes

            pipelines = []
            for l1_pipeline in l1_pipelines:
                pipeline = self.l1_to_proxy_pipeline(l1_pipeline)
                if pipeline:
                    self.pipeline_hash[str(pipeline)] = l1_pipeline
                    pipelines.append(pipeline)

            return pipelines
        except Exception as e:
            print('get_pipelines l1 proxy excpetion ', e)
            return None

    def l1_to_proxy_pipeline(self, l1_pipeline):
        pipeline = Pipeline()
        ok = True
        for prim in l1_pipeline.get_primitives():
            l2prim = self.primitive_hash.get(prim.name, None)
            if not l2prim:
                ok = False
                break
            pipeline.addPrimitive(l2prim)

        if ok:
            return pipeline
        return None

    def get_related_pipelines(self, pipeline):
        pipelines = []
        l1_pipeline = self.pipeline_hash.get(str(pipeline), None)
        if l1_pipeline:
            l1_pipelines = self.l1_planner.find_similar_learner(l1_pipeline, include_siblings=True)
            for l1_pipeline in l1_pipelines:
                pipeline = self.l1_to_proxy_pipeline(l1_pipeline)
                if pipeline:
                    self.pipeline_hash[str(pipeline)] = l1_pipeline
                    pipelines.append(pipeline)
        return pipelines
