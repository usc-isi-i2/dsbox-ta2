from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class AlphaZeroEvalTemplate(DSBoxTemplate): # this is a template from succeed pipeline for uu2 dataset
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Alpha_Zero_template",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_prediction_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps":[
                {
                    "name": "to_dataframe_step", # step 1
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "column_parser_step", # step 2
                    "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_attribute_step", # step 3
                    "primitives": [
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
                    "name": "encoder_step", # step 4
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.one_hot_encoder.SKlearn",
                            "hyperparameters":{
                                "handle_unknown": ("ignore",)
                            }
                        }
                    ],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "cast_step", # step 5
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.cast_to_type.Common",
                            "hyperparameters":{
                                "type_to_case": ("float",)
                            }
                        }
                    ],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step", # step 6
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
                    "inputs":["column_parser_step"]
                },
                {
                    "name": "cast_step_target", # step 7
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.cast_to_type.Common"
                        }
                    ],
                    "inputs": ["extract_target_step"]
                },
                {
                    "name": "model_step", # step 8
                    "primitives":[
                        {
                            "primitive": "d3m.primitives.regression.ridge.SKlearn"
                        }
                    ],
                    "inputs": ["cast_step", "cast_step_target"]
                },
                {
                    "name": "construct_prediction_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.Common",
                        }
                    ],
                    "inputs": ["model_step", "column_parser_step"]
                }
            ]
        }


