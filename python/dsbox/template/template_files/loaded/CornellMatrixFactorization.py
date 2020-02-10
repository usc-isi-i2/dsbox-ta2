from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class CornellMatrixFactorization(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Cornell_matrix_factorization",
            "taskType": {TaskKeyword.COLLABORATIVE_FILTERING.name, TaskKeyword.VERTEX_CLASSIFICATION.name, TaskKeyword.COMMUNITY_DETECTION.name, TaskKeyword.LINK_PREDICTION.name},
            "taskSubtype":  {"NONE", TaskKeyword.NONOVERLAPPING.name, TaskKeyword.OVERLAPPING.name, TaskKeyword.MULTICLASS.name, TaskKeyword.BINARY.name, TaskKeyword.MULTILABEL.name, TaskKeyword.MULTIVARIATE.name, TaskKeyword.UNIVARIATE.name},
            #"taskSubtype": "NONE",
            #"inputType": "table",
            "inputType": {"graph","table", "edgeList"},
            "output": "model_step",
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives":["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs":["template_input"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "parser_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters": {
                            "semantic_types": ('https://metadata.datadrivendiscovery.org/types/Attribute',)
                        }
                    }],
                    "inputs":["common_profiler_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives":[
                        {"primitive":"d3m.primitives.data_preprocessing.standard_scaler.SKlearn", },
                        {"primitive":"d3m.primitives.normalization.iqr_scaler.DSBOX", }
                    ],
                    "inputs": ["parser_step"]
                },
                {
                    "name": "matrix_factorization",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.collaborative_filtering.high_rank_imputer.Cornell",
                            "hyperparameters": {
                                "d": [0, 10, 20, 50, 100, 200],
                                "alpha": [0.01, 0.1, 0.5, 1.0],
                                "beta":[0.01, 0.1, 0.5, 1.0],
                                "maxiter": [200, 500, 1000, 2000]
                                }
                        }],
                    "inputs":["scaler_step", "scaler_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters": {
                                "semantic_types":("https://metadata.datadrivendiscovery.org/types/TrueTarget",)}
                }],
                    "inputs":["common_profiler_step"]
                },
                *TemplateSteps.classifier_model(feature_name="matrix_factorization",
                                                target_name='extract_target_step')
                # low rank hyperparams : k
                ]
            }

