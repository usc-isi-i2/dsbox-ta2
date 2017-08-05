from dsbox.planner.levelone.planner import (LevelOnePlanner, get_d3m_primitives, AffinityPolicy)
from primitives.library import PrimitiveLibrary

class LevelOnePlannerProxy(object):
    """
    The Level-1 DSBox Proxy Planner. 

    This is here to integrate with Ke-Thia's L1 Planner until we come up with a consistent interface
    """    
    def __init__(self, libdir):
        self.models = PrimitiveLibrary(libdir+"/models.json")
        self.features = PrimitiveLibrary(libdir+"/features.json")

        self.primitives = get_d3m_primitives()
        self.policy = AffinityPolicy(self.primitives)
        self.l1_planner = LevelOnePlanner(primitives=self.primitives, policy=self.policy)

        self.model_hash = {}
        for model in self.models.primitives:
            self.model_hash[model.name] = model

        self.pipeline_hash = {}
        
    def get_pipelines(self, problemtype, subtype, data):
        pipelines = []
        l1_pipelines = self.l1_planner.generate_pipelines_with_hierarchy(level=3)
        for l1_pipeline in l1_pipelines:
            pipeline = self.l1_to_proxy_pipeline(l1_pipeline)
            if pipeline:
                self.pipeline_hash[str(pipeline)] = l1_pipeline
                pipelines.append(pipeline)
        return pipelines

    def l1_to_proxy_pipeline(self, l1_pipeline):
        pipeline = []
        ok = True
        for prim in l1_pipeline.get_primitives():
            l2prim = self.model_hash.get(prim.name, None)
            if not l2prim:
                ok = False
                break
            pipeline.append(l2prim)

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

    def get_pipelines_old(self, problemtype, subtype, data):
        pipelines = []
        feature_to_use = None
        for feature in self.features.primitives:
            if feature.name == "TFIDF":
                feature_to_use = feature
        
        for model in self.models.primitives:
            if model.type:
                if problemtype.lower() == model.type.lower():
                    pipeline = []
                    pipeline.append(model)
                    # TODO: Add featurisation etc
                    pipelines.append(pipeline)

        return pipelines
