from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class SVRRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "svr_regression_template",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name, "NONE"},
            "taskType": TaskKeyword.REGRESSION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                            "d3m.primitives.regression.svr.SKlearn",
                            "hyperparameters":
                            {
                                'C': [0.8, 1.0, 1.2],
                                'kernel': ['rbf', 'poly'],
                                'degree': [2, 3, 4, 5],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


