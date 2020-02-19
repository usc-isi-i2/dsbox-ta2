from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class UU3TestTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "UU3_Test_Template",
            "taskSubtype": {TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "inputType": "table",
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "multi_table_processing_step",
                    "primitives": ["d3m.primitives.feature_extraction.multitable_featurization.DSBOX"],
                    "inputs": ["template_input"]
                }] + 
                TemplateSteps.dsbox_generic_steps() +
                     TemplateSteps.dsbox_feature_selector("classification",
                                                          first_input='data',
                                                          second_input='target') +
                     [
                         {
                             "name": "model_step",
                             "runtime": {
                                 "cross_validation": 5,
                                 "stratified": True
                             },
                             "primitives": [
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.random_forest.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'bootstrap': ["bootstrap", "disabled"],
                                            'max_depth': [15, 30, None],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'n_estimators': [10, 50, 100],
                                         }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.extra_trees.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'bootstrap': ["bootstrap", "disabled"],
                                            'max_depth': [15, 30, None],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'n_estimators': [10, 50, 100],
                                         }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.gradient_boosting.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'max_depth': [2, 3, 4, 5],
                                            'n_estimators': [50, 60, 80, 100],
                                            'learning_rate': [0.1, 0.2, 0.4, 0.5],
                                            'min_samples_split': [2, 3],
                                            'min_samples_leaf': [1, 2],
                                         }
                                 },
                             ],
                             "inputs": ["feature_selector_step", "target"]
                         }
                     ]
                
        }



################################################################################################################
#####################################   HorizontalTemplate   ###################################################
################################################################################################################


