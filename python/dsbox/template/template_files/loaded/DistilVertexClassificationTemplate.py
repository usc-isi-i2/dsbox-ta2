from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DistilVertexClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "weight": 100,
            "name": "Distil_Vertex_Classification_Template",
            "taskType": {TaskKeyword.VERTEX_CLASSIFICATION.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype":  {TaskKeyword.VERTEX_CLASSIFICATION.name},
            "inputType": {"edgeList", "graph"},
            "output": "prediction_step",
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "parser_step",
                    "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                        {
                            'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                    }],
                    "inputs": ["parser_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": ["d3m.primitives.data_transformation.simple_column_parser.DataFrameCommon"],
                    "inputs": ["pre_extract_target_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                        {
                            'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                    }],
                    "inputs": ["parser_step"]
                },
                {
                    "name": "imputer_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_cleaning.imputer.SKlearn",
                        "hyperparameters":
                        {
                            'return_result': ["replace"],
                            'use_semantic_types': [True],
                            'strategy': ["most_frequent"]
                        }
                    }],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encoder_step1",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.encoder.DistilTextEncoder",
                        "hyperparameters":
                        {
                        }
                    }],
                    "inputs": ["imputer_step", "extract_target_step"]
                },
                {
                    "name": "encoder_step2",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.one_hot_encoder.DistilOneHotEncoder",
                        "hyperparameters":
                        {
                        }
                    }],
                    "inputs": ["encoder_step1"]
                },
                {
                    "name": "scaler_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_preprocessing.robust_scaler.SKlearn",
                        "hyperparameters":
                        {
                        }
                    }],
                    "inputs": ["encoder_step2"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.xgboost_gbtree.Common",
                            "hyperparameters":
                            {
                                'n_jobs': [2],
                                'return_result': ["new"],
                                'add_index_columns': [True],
                                'fit_prior': [False],
                                'n_estimators': [80, 120],
                                'learning_rate':[0.1, 0.5],
                                'max_depth': [5, 10],
                                "gamma": [0.4, 0.6],
                                "min_child_weight": [1]
                                
                            }
                        }, 
                        ],
                    "inputs": ["scaler_step", "extract_target_step"]
                },
                {
                    "name": "prediction_step",
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "to_dataframe_step"]
                }
            ]
        }


