from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class JHUGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Graph_Matching_Template",
            "taskType": TaskKeyword.GRAPH_MATCHING.name,
            "taskSubtype": {"NONE", TaskKeyword.NONOVERLAPPING.name, TaskKeyword.OVERLAPPING.name, TaskKeyword.MULTICLASS.name, TaskKeyword.BINARY.name, TaskKeyword.MULTILABEL.name, TaskKeyword.MULTIVARIATE.name, TaskKeyword.UNIVARIATE.name},
            "inputType": {"edgeList", "graph"},
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.graph_matching.seeded_graph_matching.JHU"],
                    "inputs": ["template_input"]
                }
            ]
        }
################################################################################################################
#####################################   AudioClassificationTemplate   ##########################################
################################################################################################################


