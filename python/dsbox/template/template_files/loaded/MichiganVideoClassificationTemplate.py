from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class MichiganVideoClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Michigan_Video_Classification_Template",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "taskSubtype": TaskKeyword.MULTICLASS.name,
            "inputType": "video",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_prediction_step",  # Name of the final step generating the prediction
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
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "read_video_step",
                    "primitives": ["d3m.primitives.data_preprocessing.video_reader.Common"],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "featurize_step",
                    "primitives": ["d3m.primitives.feature_extraction.i3d.Umich"],
                    "inputs": ["read_video_step"]

                },
                {
                    "name": "convert_step",
                    "primitives": ["d3m.primitives.data_transformation.ndarray_to_dataframe.Common"],
                    "inputs": ["featurize_step"]

                },
                {
                    "name": "model_step",
                    # "primitives": ["d3m.primitives.classifier.RandomForest"],
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }],
                    "inputs": ["convert_step", "extract_target_step"]
                },
                {
                    "name": "construct_prediction_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.Common",
                        }
                    ],
                    "inputs": ["model_step", "to_dataframe_step"]
                }
            ]
        }


################################################################################################################
#####################################   TA1Template   ##########################################################
################################################################################################################

