from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class JHULinkPredictionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Link_Prediction_Template",
            "taskType": {TaskKeyword.GRAPH_MATCHING.name, TaskKeyword.GRAPH.name, TaskKeyword.LINK_PREDICTION.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype":  {"NONE", TaskKeyword.GRAPH_MATCHING.name, TaskKeyword.GRAPH.name, TaskKeyword.LINK_PREDICTION.name},
            "inputType": {"graph"},
            "output": "model_step",
            "steps": [
                {
                    "name": "readgraph_step1",
                    "primitives": [
                        "d3m.primitives.link_prediction.data_conversion.JHU"
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "embedding_step",
                    "primitives": [
                        "d3m.primitives.data_transformation.laplacian_spectral_embedding.JHU", 
                        {
                            "primitive": "d3m.primitives.data_transformation.adjacency_spectral_embedding.JHU",
                            "hyperparameters": {
                                "max_dimension": [5],
                                "use_attributes": [True]
                            }
                        }
                    ],
                    "inputs": ["readgraph_step2"]

                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": 
                            "d3m.primitives.link_prediction.rank_classification.JHU",
                            "hyperparameters": {
                            }
                        }
                    ],
                    "inputs": ["embedding_step"]
                }
            ]
        }



