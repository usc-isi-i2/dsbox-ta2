from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
import multiprocessing
CURRENT_CPU_COUNT = multiprocessing.cpu_count()

class DefaultImageProcessingClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DefaultImageProcessingClassificationTemplate",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            # See TaskKeyword, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_predictions_step",  # Name of the final step generating the prediction
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
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                        "hyperparameters": {
                            "drop_non_numeric_columns": [False]
                        }
                    }],
                    "inputs": ["pre_extract_target_step"]
                },
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.resnet50_image_feature.DSBOX",
                            "hyperparameters": {
                                'generate_metadata': [True]
                            }
                        },
                        {
                            "primitive": "d3m.primitives.feature_extraction.vgg16_image_feature.DSBOX",
                            "hyperparameters": {
                                'generate_metadata': [True]
                            }
                        }

                    ],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.pca.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types': [True]
                            }
                        },
                        "d3m.primitives.data_preprocessing.do_nothing.DSBOX"
                    ],
                    "inputs": ["feature_extraction"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.linear_discriminant_analysis.SKlearn",
                            "hyperparameters": {
                            }
                        },
                        {
                            "primitive": "d3m.primitives.classification.passive_aggressive.SKlearn",
                            "hyperparameters": {}
                        },
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                'bootstrap': ["bootstrap", "disabled"],
                                'max_depth': [15, 30, None],
                                'min_samples_leaf': [1, 2, 4],
                                'min_samples_split': [2, 5, 10],
                                'max_features': ['auto', 'sqrt'],
                                'n_estimators': [10, 50, 100]
                            }
                        },
                        {
                            "primitive": "d3m.primitives.classification.xgboost_gbtree.Common",
                            "hyperparameters": {
                                # 'learning_rate': [0.001, 0.1],
                                # 'max_depth': [15, 30, None],
                                # # 'min_samples_leaf': [1, 2, 4],
                                # # 'min_samples_split': [2, 5, 10],
                                # 'n_more_estimators': [10, 50, 100, 1000],
                                # 'n_estimators': [10, 50, 100, 1000],
                                "n_jobs": [int(CURRENT_CPU_COUNT * 0.9)]
                            }
                        },
                        {
                            "primitive": "d3m.primitives.classification.xgboost_dart.Common",
                            "hyperparameters": {
                                "n_jobs": [int(CURRENT_CPU_COUNT * 0.9)]
                            }
                        }
                    ],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
                {
                    "name": "construct_predictions_step",#step 7
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "to_dataframe_step"]
                }
            ]
        }

