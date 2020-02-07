from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class ClassificationWithSelection(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "classification_with_feature_selection",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.human_steps() + TemplateSteps.dsbox_feature_selector("classification") + [
                {
                    "name": "model_step",
                    "primitives": [
                        # {
                        #     "primitive": "d3m.primitives.classification.sgd.SKlearn",
                        #     "hyperparameters": {
                        #         'use_semantic_types': [True],
                        #         'return_result': ['new'],
                        #         'add_index_columns': [True],
                        #         "loss": ['log', 'hinge', 'squared_hinge', 'perceptron'],
                        #         "alpha": [float(x) for x in np.logspace(-6, -1.004, 7)],
                        #         "l1_ratio": [float(x) for x in np.logspace(-9, -0.004, 7)],
                        #         "penalty": ['elasticnet', 'l2']
                        #     }
                        # },
                        {
                            "primitive":
                            "d3m.primitives.classification.gradient_boosting.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                'max_depth': [2, 5],
                                'n_estimators': [50, 100],
                                'learning_rate': [0.1, 0.3],
                                'min_samples_split': [2, 3],
                                'min_samples_leaf': [1, 2],
                            }
                        },

                             ],
                    "inputs": ["feature_selector_step", "extract_target_step"]
                }
            ]
        }


