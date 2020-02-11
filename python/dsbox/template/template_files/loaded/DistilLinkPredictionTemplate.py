from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DistilLinkPredictionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Distil_LinkPrediction_Template",
            "taskType": {TaskKeyword.LINK_PREDICTION.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype":  {TaskKeyword.LINK_PREDICTION.name},
            "inputType": {"graph"},
            "output": "predict_step",
            "steps": [
                {
                    "name": "parse_step",
                    "primitives": ["d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader"],
                    "inputs":['template_input']
                    },
                {
                    "name": "predict_step",
                    "primitives":[
                        {
                            "primitive": "d3m.primitives.link_prediction.link_prediction.DistilLinkPrediction",
                            "hyperparameters": {
                                "metric": [("accuracy"),]
                            }
                        }
                    ],
                    "inputs":["parse_step", "parse_step_produce_target"]
                },
            ]
        }


