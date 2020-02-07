from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DefaultObjectDetectionTemplateFittedWeight(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DefaultObjectDetectionTemplateFittedWeight",
            "taskType": {TaskKeyword.OBJECT_DETECTION.name},
            "taskSubtype": {TaskKeyword.OBJECT_DETECTION.name},
            "inputType": {"table", "image"},  # See SEMANTIC_TYPES.keys() for range of values
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
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
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
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step", # step 4
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.yolo.DSBOX",
                            "hyperparameters": {
                                "use_fitted_weight": [(True)],
                            }
                        }
                    ],
                    "inputs": ["extract_file_step", "extract_target_step"]
                },
            ]
        }


