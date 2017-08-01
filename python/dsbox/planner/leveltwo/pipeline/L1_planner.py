from primitives.library import PrimitiveLibrary

class L1_Planner(object):
    """
    The Level-1 DSBox Dummy Planner. 

    This is here till it is replaced with Ke-Thia's actual L1 Planner
    """    
    def __init__(self, libdir):
        self.models = PrimitiveLibrary(libdir+"/models.json")
        self.features = PrimitiveLibrary(libdir+"/features.json")
        
    def get_pipelines(self, problemtype, subtype, data):
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