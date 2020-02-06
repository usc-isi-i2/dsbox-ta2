from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DistilAudioClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DistilAudioClassificationTemplate",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": "audio",
            "output": "construct_predictions_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "sampler_step",
                    "primitives" :[
                        {
                            "primitive": "d3m.primitives.data_preprocessing.dataset_sample.Common",
                            "hyperparameters" :{
                                "sample_size": [0.75]
                            }
                        }
                    ],
                    "inputs" : ["template_input"]
                },
                {
                    "name": "dataset_loader_step",
                    "primitives": ["d3m.primitives.data_preprocessing.audio_reader.DistilAudioDatasetLoader"],
                    "inputs": ["sampler_step"]
                },
                {
                    "name": "column_parser_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.column_parser.Common",
                            "hyperparameters": {
                                "parse_semantic_types": [(
                                    "http://schema.org/Boolean",
                                    "http://schema.org/Integer",
                                    "http://schema.org/Float",
                                    "https://metadata.datadrivendiscovery.org/types/FloatVector")
                                ],
                            }

                        }],
                    "inputs": ["dataset_loader_step"]
                },
                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': (
                                "https://metadata.datadrivendiscovery.org/types/Target",
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["column_parser_step"]
                },
                # read X value
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.feature_extraction.audio_transfer.DistilAudioTransfer",
                        "hyperparameters":
                            {}
                    }],
                    "inputs": ["dataset_loader_step_produce_collection"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.learner.random_forest.DistilEnsembleForest",
                            "hyperparameters": {
                            "metric": [("accuracy")]
                            }
                        },
                    ],
                    "inputs": ["extract_attribute_step", "extract_target_step"]
                },
                {
                    "name": "construct_predictions_step",#step 7
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "column_parser_step"]
                }
            ]
        }


################################################################################################################
#####################################   ClusteringTemplate   ###################################################
################################################################################################################


