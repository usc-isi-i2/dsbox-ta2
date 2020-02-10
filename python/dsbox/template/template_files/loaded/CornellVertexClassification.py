from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 

# may not work because of different hyperparamters for "encoder/decoder step"
class CornellVertexClassification(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Cornell_vertex_classification",
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
                    "name": "preprocess_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.column_parser.preprocess_categorical_columns.Cornell",
                        "hyperparameters": {
                        }
                    }],
                    "inputs":["parser_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.data_preprocessing.standard_scaler.SKlearn",
                        "hyperparameters": {
                        }
                    }],
                    "inputs":["preprocess_step"]
                },
                {
                    "name": "pca_step",
                    "primitives":[
                    {
                        "primitive":"d3m.primitives.feature_extraction.huber_pca.Cornell",
                        "hyperparameters": {
                            "alpha": [0.1],
                            "d": [15, 20],
                            "epsilon": [0.1],
                            "maxiter": [1000, 5000],
                            "t": [0.001]
                        },
                    },
                    {
                        "primitive":"d3m.primitives.collaborative_filtering.high_rank_imputer.Cornell",
                        "hyperparameters": {
                            "alpha": [0.1],
                            "beta": [0.01],
                            "d": [15, 20],
                            "epsilon": [0.1],
                            "maxiter": [1000, 5000]
                        },
                    }],
                    "inputs":["scaler_step"]
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
                {
                    "name": "encoder_targer_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.data_preprocessing.label_encoder.Common",
                        "hyperparameters": {
                        }
                    }],
                    "inputs":["extract_target_step"]
                },
                {
                    "name": "model_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.classification.svc.SKlearn",
                        "hyperparameters": {
                        }
                    }],
                    "inputs":["scaler_step", "encoder_targer_step"]
                },
                {
                    "name": "decoder_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.data_preprocessing.label_decoder.Common",
                        "hyperparameters": {
                            "encoder": [7]
                        }
                    }],
                    "inputs":["model_step"]
                },
                {
                    "name": "construct_prediction_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.data_transformation.construct_predictions.Common",
                        "hyperparameters": {
                        }
                    }],
                    "inputs":["decoder_step", "to_dataframe_step"]
                }
                ]
            }

