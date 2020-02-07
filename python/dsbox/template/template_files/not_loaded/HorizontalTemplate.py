from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class HorizontalTemplate(DSBoxTemplate): #This template only generate processed features
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Horizontal_Template",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name, TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": {TaskKeyword.CLASSIFICATION.name, TaskKeyword.REGRESSION.name},
            "inputType": "table",
            "output": "scaler_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                                    ),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": [
                        "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                    ],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_construction.corex_text.DSBOX",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": [
                        "d3m.primitives.data_preprocessing.encoder.DSBOX",
                    ],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.normalization.iqr_scaler.DSBOX",
                            "hyperparameters": {}
                        },
                    ],
                    "inputs": ["impute_step"]
                },
                # {
                #     "name": "extract_target_step",
                #     "primitives": [{
                #         "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                #         "hyperparameters":
                #             {
                #                 'semantic_types': (
                #                     #'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                #                     'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                #                 'use_columns': (),
                #                 'exclude_columns': ()
                #             }
                #     }],
                #     "inputs": ["to_dataframe_step"]
                # },
            ]
        }

