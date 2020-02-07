from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DataAugmentRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)

        self.template = {
            "name": "data_augment_regression_template",
            "taskType": TaskKeyword.REGRESSION.name,
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name, "NONE"},
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
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
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["common_profiler_step"]
            },
            {
                "name": "extract_target_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "target_process_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["extract_target_step"]
            },
            {
                "name": "profile_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX"
                ],
                "inputs": ["profile_step"]
            },
            {
                "name": "encode_text_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.feature_construction.corex_text.DSBOX",
                        "hyperparameters":
                            {
                                'n_hidden': [(10)],
                                'threshold': [(500)],
                                'n_grams': [(1)],
                                'max_df': [(.9)],
                                'min_df': [(.02)],
                            }
                    },
                ],
                "inputs": ["clean_step"]
            },
            {
                "name": "encoder_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.label_encoder.DSBOX"
                ],
                "inputs": ["encode_text_step"]
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                "inputs": ["encode_text_step"]
            },
            {
                "name": "model_step",
                "primitives": [
                    {
                        "primitive" : "d3m.primitives.regression.gradient_boosting.SKlearn",
                        "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                            }
                    },
                ],
                "inputs": ["encode_text_step", "target_process_step"]
            },


            ]
        }


