from primitives.library import PrimitiveLibrary

class L1_Planner(object):
    def __init__(self, libdir):
        self.models = PrimitiveLibrary(libdir+"/models.json")
        self.features = PrimitiveLibrary(libdir+"/features.json")
        
    def get_pipelines(self, problemtype, data):
        pipelines = []
        for model in self.models.primitives:
            for typ in model.types:
                if problemtype.lower() == typ.lower():
                    pipeline = []
                    pipeline.append(model)
                    # TODO: Add featurisation etc
                    pipelines.append(pipeline)
                    break
        return pipelines