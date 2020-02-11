from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DistilGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Distil_GraphMatching_Template",
            "taskType": {TaskKeyword.GRAPH_MATCHING.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype":  {TaskKeyword.GRAPH_MATCHING.name},
            "inputType": {"graph"},
            "output": "predict_step",
            "steps": [
                {
                    "name": "parse_step",
                    "primitives": ["d3m.primitives.data_transformation.load_graphs.DistilGraphLoader"],
                    "inputs":['template_input']
                    },
                {
                    "name": "predict_step",
                    "primitives":[
                        {
                            "primitive": "d3m.primitives.graph_matching.seeded_graph_matching.DistilSeededGraphMatcher",
                            "hyperparameters": {
                                "metric": [("accuracy"),]
                            }
                        }
                    ],
                    "inputs":["parse_step", "parse_step_produce_target"]
                },
            ]
        }


