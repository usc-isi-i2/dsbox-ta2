from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class dsboxRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "dsbox_regression_template",
            "taskType": TaskKeyword.REGRESSION.name,
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name, "NONE"},
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                *TemplateSteps.dsbox_preprocessing(
                    clean_name="clean_step",
                    target_name="extract_target_step"
                ),
                *TemplateSteps.dsbox_encoding(
                    clean_name="clean_step",
                    encoded_name="encoder_step"
                ),
                *TemplateSteps.dsbox_imputer(
                    encoded_name="encoder_step",
                    impute_name="impute_step"
                ),
                *TemplateSteps.regression_model(
                    feature_name="impute_step",
                    target_name="extract_target_step"
                ),
            ]
        }



################################################################################################################
#####################################   Templates for Large Datasets  ##########################################
################################################################################################################


