from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class LupiRfClassification(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "LupiRfClassification",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": {"table"},
            "specializedProblem": {SpecializedProblem.PRIVILEGED_INFORMATION},
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
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive":  "d3m.primitives.classification.lupi_rf.LupiRFClassifier",
                            "hyperparameters": {
                                "add_index_columns": [False],
                                "bootstrap": [True],
                                "class_weight": ["balanced"],
                                "criterion": ["gini"],
                                "error_on_no_input": [True],
                                "exclude_input_columns": [()],
                                "exclude_output_columns": [()],
                                "max_depth": [10],
                                "max_features": ["auto"],
                                "max_leaf_nodes": [None],
                                "min_impurity_decrease": [0.0],
                                "min_samples_leaf": [1],
                                "min_samples_split": [2],
                                "min_weight_fraction_leaf": [0],
                                "n_estimators": [5000],
                                "n_jobs": [4],
                                "oob_score": [False],
                                "return_result": ["new"],
                                "use_input_columns": [()],
                                "use_output_columns": [()],
                                "use_semantic_types": [False],
                                "warm_start": [False],
                            }
                        },
                        {
                            "primitive":  "d3m.primitives.classification.lupi_rfsel.LupiRFSelClassifier",
                            "hyperparameters":
                            {
                                "add_index_columns": [False],
                                "bootstrap": [True],
                                "class_weight": ["balanced"],
                                "criterion": ["gini"],
                                "error_on_no_input": [True],
                                "exclude_input_columns": [()],
                                "exclude_output_columns": [()],
                                "max_depth": [20],
                                "max_features": ["auto"],
                                "max_leaf_nodes": [None],
                                "min_impurity_decrease": [0.0],
                                "min_samples_leaf": [1],
                                "min_samples_split": [2],
                                "min_weight_fraction_leaf": [0],
                                "n_cv": [5],
                                "n_estimators": [2000],
                                "n_jobs": [4],
                                "oob_score": [False],
                                "regressor_type": ["linear"],
                                "return_result": ["new"],
                                "use_input_columns": [()],
                                "use_output_columns": [()],
                                "use_semantic_types": [False],
                                "warm_start": [False],

                            }
                        },
                    ],
                    "inputs": ["common_profiler_step", "extract_target_step"]
                },
                {
                    "name": "prediction_step",
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "common_profiler_step"]
                }

            ]
        }

