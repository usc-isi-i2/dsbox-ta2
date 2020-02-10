from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DistilPreprocessingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Distil_preprocessing_problem_template",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": {"table"},
            "output": "prediction_step",
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
                    "name": "parser_step",
                    "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_target_step",
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
                    "name": "data_clean_step1",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.data_cleaning.DistilEnrichDates",
                        "hyperparameters":
                        {
                        }
                    }],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "data_clean_step2",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.data_cleaning.DistilReplaceSingletons",
                        "hyperparameters":
                        {
                        }
                    }],
                    "inputs": ["data_clean_step1"]
                },
                {
                    "name": "imputer_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.imputer.DistilCategoricalImputer",
                        "hyperparameters":
                        {
                        }
                    }],
                    "inputs": ["data_clean_step2"]
                },
                {
                    "name": "encoder_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.encoder.DistilTextEncoder",
                        "hyperparameters":
                        {
                        }
                    }],
                    "inputs": ["imputer_step", "extract_target_step"]
                },
                {
                    "name": "one_hot_encoder_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.one_hot_encoder.DistilOneHotEncoder",
                        "hyperparameters":
                        {
                            "max_one_hot":[16],
                        }
                    }],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "binary_encoder_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.encoder.DistilBinaryEncoder",
                        "hyperparameters":
                        {
                            "min_binary":[17],
                        }
                    }],
                    "inputs": ["one_hot_encoder_step"]
                },
                {
                    "name": "missing_indicator_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_cleaning.missing_indicator.SKlearn",
                        "hyperparameters":
                        {
                            "use_semantic_types":[True],
                            "return_result":["append"],
                            "error_on_new":[False],
                            "error_on_no_input":[False]
                        }
                    }],
                    "inputs": ["binary_encoder_step"]
                },
                {
                    "name": "imputer_step2",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_cleaning.imputer.SKlearn",
                        "hyperparameters":
                        {
                            "use_semantic_types":[True],
                            "return_result":["replace"],
                        }
                    }],
                    "inputs": ["missing_indicator_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                            "d3m.primitives.classification.gradient_boosting.SKlearn",
                            "hyperparameters":
                            {
                                'max_depth': [2, 5],
                                'n_estimators': [50, 100],
                                'learning_rate': [0.1, 0.3],
                                'min_samples_split': [2, 3],
                                'min_samples_leaf': [1, 2],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                        {
                            "primitive":
                            "d3m.primitives.classification.multinomial_naive_bayes.SKlearn",
                            "hyperparameters":
                            {
                                'alpha': [0, .5, 1],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                        {
                            "primitive":
                            "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                    ],
                    "inputs": ["imputer_step2", "extract_target_step"]
                },
                {
                    "name": "prediction_step",
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "common_profiler_step"]
                }

            ]
        }



