from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class AlternativeClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "alternative_classification_template",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_predictions_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() +
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
                                 "d3m.primitives.classification.logistic_regression.SKlearn",
                                 "d3m.primitives.classification.linear_discriminant_analysis.SKlearn",
                                 "d3m.primitives.classification.passive_aggressive.SKlearn",
                                 "d3m.primitives.classification.k_neighbors.SKlearn",
                             ],
                             "inputs": ["feature_selector_step", "target"]
                         },
                         {
                                "name": "construct_predictions_step",#step 7
                                "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                                "inputs": ["model_step", "to_dataframe_step"]
                         }

                     ]
        }