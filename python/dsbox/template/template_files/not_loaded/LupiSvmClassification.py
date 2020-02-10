from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class LupiSvmClassification(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "LupiSvmClassification",
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
                    "name": "model_step",
                    "primitives": [{
                        "primitive":  "d3m.primitives.classification.lupi_svm.LupiSvmClassifier",
                        "hyperparameters": {
                            "C": [1],
                            "C_gridsearch": [(-4.0, 26.0, 0.3)],
                            "add_index_columns": [False],
                            "class_weight": ['balanced'],
                            "coef0": [0],
                            "degree": [3],
                            "gamma": ["auto"],
                            "gamma_gridsearch": [(-4.0, 26.0, 0.3)],
                            "kernel": ["rbf"],
                            "max_iter": [-1],
                            "n_jobs": [4],
                            "probability": [False],
                            "return_result": ["new"],
                            "shrinking": [True],
                            "tol": [0.001],
                            "use_semantic_types": [False],
                        }
                    },
                    ],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "prediction_step",
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "common_profiler_step"]
                }

            ]
        }

