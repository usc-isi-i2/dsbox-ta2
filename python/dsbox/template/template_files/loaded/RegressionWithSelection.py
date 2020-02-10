from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class RegressionWithSelection(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "regression_with_feature_selection",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.human_steps() + TemplateSteps.dsbox_feature_selector("regression") +
                     [
                         {
                             "name": "model_step",
                             "primitives": [
                                 {
                                     "primitive": "d3m.primitives.regression.sgd.SKlearn",
                                     "hyperparameters": {
                                         "loss": ['squared_loss', 'huber'],
                                         "alpha": [float(x) for x in np.logspace(-5, -1.004, 7)],  # cannot reach 0.1
                                         "l1_ratio": [0.01, 0.15, 0.3, 0.5, 0.6, 0.7, 0.9],  # cannot reach 1
                                         "learning_rate": ['optimal', 'invscaling'],
                                         'add_index_columns': [True],
                                         'use_semantic_types':[True],
                                     }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.regression.gradient_boosting.SKlearn",
                                     "hyperparameters":
                                         {
                                             'max_depth': [2, 3, 5],
                                             'n_estimators': [100, 150, 200],
                                             'learning_rate': [0.1, 0.3, 0.5],
                                             'min_samples_split': [2, 3],
                                             'min_samples_leaf': [1, 2],
                                             'add_index_columns': [True],
                                             'use_semantic_types':[True],
                                         }
                                 },
                             ],
                             "inputs": ["feature_selector_step", "extract_target_step"]
                         }
                     ]
        }


