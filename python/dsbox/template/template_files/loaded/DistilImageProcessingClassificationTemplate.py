from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
import multiprocessing
CURRENT_CPU_COUNT = multiprocessing.cpu_count()

class DistilImageProcessingClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DistilImageProcessingClassificationTemplate",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            # See TaskKeyword, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_predictions_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "sampler_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_preprocessing.dataset_sample.Common",
                        "hyperparameters":
                            {
                                'sample_size':[10000],
                            }
                    }],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["sampler_step"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "column_parser_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.column_parser.Common",
                        "hyperparameters":
                            {
                                'parse_semantic_types':[(
                                    "http://schema.org/Boolean",
                                    "http://schema.org/Integer",
                                    "http://schema.org/Float",
                                    "https://metadata.datadrivendiscovery.org/types/FloatVector"
                                  )],
                            }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.digital_image_processing.convolutional_neural_net.Gator",
                            "hyperparameters": {
                                'unfreeze_proportions': [0.5]
                            }
                        },
                    ],
                    "inputs": ["column_parser_step", "extract_target_step"]
                },
                {
                    "name": "construct_predictions_step",#step 7
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "common_profiler_step"]
                }
            ]
        }

