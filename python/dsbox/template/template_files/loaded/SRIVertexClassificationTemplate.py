from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class SRIVertexClassificationTemplate(DSBoxTemplate):
    # not used for DS01876
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Vertex_classification_Template",
            "taskType": {TaskKeyword.VERTEX_CLASSIFICATION.name},
            "taskSubtype":  {TaskKeyword.VERTEX_CLASSIFICATION.name},
            #"taskType": TaskKeyword.VERTEX_NOMINATION.name,
            #"taskSubtype": "NONE",
            "inputType": {"graph", "edgeList", "table"},
            "output": "model_step",
            "steps": [
                {
                    "name": "parse_step",
                    "primitives": ["d3m.primitives.data_transformation.vertex_classification_parser.VertexClassificationParser"],
                    "inputs": ["template_input"]

                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.classification.vertex_nomination.VertexClassification"],
                    #"primitives": ["d3m.primitives.classification.community_detection.CommunityDetection"],
                    "inputs": ["parse_step"]

                }
            ]
        }


