from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class SRIGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_GraphMatching_Template",
            "taskType": {TaskKeyword.GRAPH_MATCHING.name, TaskKeyword.GRAPH.name, TaskKeyword.LINK_PREDICTION.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype":  {"NONE", TaskKeyword.GRAPH_MATCHING.name, TaskKeyword.GRAPH.name, TaskKeyword.LINK_PREDICTION.name},
            "inputType": {"graph"},
            "output": "predict_step",
            "steps": [
                {
                    "name": "parse_step",
                    "primitives": ["d3m.primitives.data_transformation.graph_matching_parser.GraphMatchingParser"],
                    "inputs":['template_input']
                    },
                {
                    "name": "transform_step",
                    "primitives":["d3m.primitives.data_transformation.graph_transformer.GraphTransformer"],
                    "inputs":["parse_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                          "primitive": "d3m.primitives.link_prediction.link_prediction.LinkPrediction",
                          "hyperparameters":
                            {
                                "prediction_column": [('match')],
                                }
                        },
                        {
                            "primitive": "d3m.primitives.link_prediction.graph_matching_link_prediction.GraphMatchingLinkPrediction",
                            "hyperparameters": {
                                "link_prediction_hyperparams": [(TemplateSteps.class_hyperparameter_generator(
                                    "d3m.primitives.link_prediction.graph_matching_link_prediction.GraphMatchingLinkPrediction", "link_prediction_hyperparams",
                                    {"truth_threshold": 0.0000001, "psl_options": "", "psl_temp_dir": "/tmp/psl/run",
                                     "postgres_db_name": "psl_d3m", "admm_iterations": 1000, "max_threads": 0,
                                     "jvm_memory": 0.75, "prediction_column": "match"}))]
                            }
                        }
                    ],
                    "inputs": ["transform_step", "transform_step"]
                },
                {
                    "name":"predict_step",
                    "primitives":["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs":["model_step"]
                }
            ]
        }



