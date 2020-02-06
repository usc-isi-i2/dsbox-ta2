from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class JHUGraphTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_ALL_IN_ONE_Graph_Template",
            "taskType": {TaskKeyword.COMMUNITY_DETECTION.name, TaskKeyword.GRAPH_MATCHING.name, TaskKeyword.LINK_PREDICTION.name, TaskKeyword.VERTEX_NOMINATION.name, TaskKeyword.VERTEX_CLASSIFICATION.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype":  {TaskKeyword.COMMUNITY_DETECTION.name, TaskKeyword.GRAPH_MATCHING.name, TaskKeyword.LINK_PREDICTION.name, TaskKeyword.VERTEX_NOMINATION.name, TaskKeyword.VERTEX_CLASSIFICATION.name},
            "inputType": {"graph"},
            "output": "model_step",
            "steps": [
                {
                    "name": "readgraph_step1",
                    "primitives": [
                        "d3m.primitives.data_transformation.load_graphs.JHU"
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "readgraph_step2",
                    "primitives": [
                        "d3m.primitives.data_preprocessing.largest_connected_component.JHU"
                    ],
                    "inputs": ["readgraph_step1"]
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
                            "d3m.primitives.graph_clustering.gaussian_clustering.JHU",
                            "hyperparameters": {
                                "max_clusters": [10]
                            }
                        },
                        {
                            "primitive": "d3m.primitives.classification.gaussian_classification.JHU",
                            "hyperparameters": {

                            }
                        }
                    ],
                    "inputs": ["embedding_step"]
                }
            ]
        }


