from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DefaultTimeseriesCollectionWithSelectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_Timeseries_Collection_With_Selection_Template",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": {"timeseries", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "random_forest_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
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
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
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
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                #     "inputs": ["extract_target_step"]
                # },

                # read X value
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.data_preprocessing.time_series_to_list.DSBOX"],
                    "inputs": ["common_profiler_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX",
                            "hyperparameters":{
                                'generate_metadata':[True],
                            }
                        }
                    ],
                    "inputs": ["timeseries_to_list_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_preprocessing.robust_scaler.SKlearn",
                            "hyperparameters":{
                                'return_result':["replace"],
                            }
                        },
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX"
                    ],
                    "inputs": ["random_projection_step"]
                },
                {
                    "name": "feature_selection_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_selection.variance_threshold.SKlearn",
                            "hyperparameters":{
                                'threshold':[0.0],
                            }
                        },
                        {
                            "primitive": "d3m.primitives.feature_selection.select_percentile.SKlearn",
                            "hyperparameters":{
                                "score_func": ["f_classif"],
                                "percentile": [24, 30, 40, 50, 60, 70],
                                "use_semantic_types": [True],
                                "error_on_no_input": [False]
                            }
                        }
                    ],
                    "inputs": ["scaler_step", "extract_target_step"]
                },
                {
                    "name": "random_forest_step",
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
                            "primitive": "d3m.primitives.classification.gradient_boosting.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                'max_depth': [2, 3, 5],
                                'learning_rate': [0.1, 0.3, 0.5],
                                'min_samples_split': [2, 3],
                                'min_samples_leaf': [1, 2],
                                'max_features': ['auto', 'sqrt', "7"],
                                'n_estimators': [10, 50, 100, 150, 200],
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
                        "d3m.primitives.classification.bagging.SKlearn",
                        "d3m.primitives.classification.logistic_regression.SKlearn",
                        {
                            "primitive": "d3m.primitives.classification.k_neighbors.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                'p': [1, 2],
                                'n_neighbors': [1, 2,5],
                                'weights': ['uniform', 'distance'],
                            }
                        },
                        {
                            "primitive": "d3m.primitives.classification.linear_svc.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                'tol': [0.0001, 0.001],
                                'penalty': ["l1", "l2"],
                                'dual': [True, False],
                                "C": [1.0, 0.5],
                                "loss": ['hinge', 'squared_hinge']
                            }
                        },
                        {
                            "primitive": "d3m.primitives.classification.svc.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                'tol': [0.0001, 0.001],
                                "probability": [True, False],
                                "shrinking": [True, False],
                                "cache_size": [100.00, 200.00],
                                'decision_function_shape': ["ovr"],
                                "C": [1.0, 0.5, 2.0, 4.0, 8.0],
                            }
                        }
                    ],
                    "inputs": ["random_projection_step", "extract_target_step"]
                },
            ]
        }


