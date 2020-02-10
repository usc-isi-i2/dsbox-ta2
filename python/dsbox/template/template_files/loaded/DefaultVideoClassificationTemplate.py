from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DefaultVideoClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DefaultVideoClassificationTemplate",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "taskSubtype": TaskKeyword.MULTICLASS.name,
            "inputType": "video",
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",#step 1
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                # read X value
                {
                    "name": "extract_file_step",#step 2
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/FileName',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_target_step",# step 3
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
                    "name": "video_reader",#step 4
                    "primitives": ["d3m.primitives.data_preprocessing.video_reader.Common"],
                    "inputs": ["extract_file_step"]
                },
                {
                    "name": "video_feature_extract",#step 5
                    "primitives": [
                            {
                                "primitive": "d3m.primitives.feature_extraction.inceptionV3_image_feature.DSBOX",
                                "hyperparameters": {
                                    "use_limitation":[(True)],
                                }
                            }

                        ],
                    "inputs": ["video_reader"]
                },
                {
                    "name": "model_step", # step 6
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.lstm.DSBOX",
                            "hyperparameters": {
                                "LSTM_units":[2048],
                                "epochs":[100, 500, 1000],
                            }
                        }
                    ],
                    "inputs": ["video_feature_extract", "extract_target_step"]
                },
            ]
        }


################################################################################################################
#####################################   TimeSeriesProblemsTemplates   ##########################################
################################################################################################################


