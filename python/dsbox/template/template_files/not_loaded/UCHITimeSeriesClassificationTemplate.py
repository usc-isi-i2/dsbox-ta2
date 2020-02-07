from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class UCHITimeSeriesClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "UCHI_Time_Series_Classification_Template",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": "timeseries",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.datasmash.d3m_XG2"],
                    "inputs": ["template_input", "template_input"]
                }
            ]
        }

################################################################################################################
#####################################   ImageProblemsTemplates   ###############################################
################################################################################################################


