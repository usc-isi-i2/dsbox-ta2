from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class JHUVertexNominationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Vertex_Nomination_Template",
            "taskType": TaskKeyword.VERTEX_CLASSIFICATION.name,
            "taskSubtype": "NONE",
            "inputType": {"edgeList", "graph"},
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
                        "d3m.primitives.data_transformation.laplacian_spectral_embedding.JHU"
                    ],
                    "inputs": ["readgraph_step2"]

                },
                {
                    "name": "model_step",
                    "primitives": [
                        "d3m.primitives.classification.gaussian_classification.JHU"
                    ],
                    "inputs": ["embedding_step"]
                }
            ]
        }

