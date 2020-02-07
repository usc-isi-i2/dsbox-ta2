from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class TESTINGTemplate(DSBoxTemplate): # this is a template from succeed pipeline for uu3 dataset
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Testing_template",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_prediction_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "update_semantic_step", # step 0
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.update_semantic_types.DatasetCommon",
                            "hyperparameters": {
                                "add_columns": [(1), (2), (3), (4), (5)],
                                "add_tpyes": ("https://metadata.datadrivendiscovery.org/types/CategoricalData",),
                                "resource_id": ("learningData", )
                            }
                        }
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step", # step 2
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
                        }
                    ],
                    "inputs": ["update_semantic_step"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "column_parser_step", # step 3
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.column_parser.Common"
                        }
                    ],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_attribute_step", # step 4
                    "primitives":[
                        {
                            "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                            "hyperparameters": {
                                    'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                    'use_columns': (),
                                    'exclude_columns': ()
                                }
                        }
                    ],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "extract_target_step", # step 5
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                            "hyperparameters": {
                                    'semantic_types': (
                                        'https://metadata.datadrivendiscovery.org/types/Target',
                                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'
                                    ),
                                    'use_columns': (),
                                    'exclude_columns': ()
                                }
                        }
                    ],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "impute_step", # step 6
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_cleaning.imputer.SKlearn",
                        }
                    ],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "model_step", # step 7
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters": {
                                "return_result": ("replace", ),
                                "use_semanctic_types": [(True)],
                            }
                        }
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                },
                {
                    "name": "construct_predict_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.Common",
                        }
                    ],
                    "inputs": ["model_step", "to_dataframe_step"]
                }

            ]
        }
################################################################################################################
#####################################   TimeSeriesForcasting Templates  ########################################
################################################################################################################


