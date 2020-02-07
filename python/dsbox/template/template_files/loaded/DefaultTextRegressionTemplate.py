from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DefaultTextRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_text_regression_template",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "inputType": {"text", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_text_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                    },
                    "primitives": [
                        {
                            "primitive":
                            "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters":
                            {
                                'max_depth': [2, 5],
                                'n_estimators': [100, 200],
                                'learning_rate': [0.1, 0.3],
                                'min_samples_split': [2, 3],
                                'min_samples_leaf': [1, 2],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                        {
                            "primitive":
                            "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


################################################################################################################
#####################################   GraphProblemsTemplates   ###############################################
################################################################################################################


