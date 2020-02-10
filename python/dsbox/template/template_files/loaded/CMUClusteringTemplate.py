from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class CMUClusteringTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.need_add_reference = True
        self.template = {
            "name": "CMU_Clustering_Template",
            "taskType": TaskKeyword.CLUSTERING.name,
            "taskSubtype": "NONE",
            "inputType": "table",
            "output": "output_step",
            "steps": [
                {
                    "name": "to_dataframe_step", # step 0
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "column_parser_step",# step 1
                    "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                    "inputs":["common_profiler_step"],
                },

                {
                    "name": "extract_attribute_step", # step 2
                    "primitives": ["d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common"],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "model_step", # step 3
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.cover_tree.Fastlvm",
                            "hyperparameters": {
                                "k": [(1), (4), (6), (8), (10), (12)]
                            }
                        }
                    ],
                    "inputs": ["data_clean_step"]
                },
                {
                    "name": "output_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.Common",
                        }
                    ],
                    "inputs": ["model_step", "column_parser_step"]
                }
            ]
        }


################################################################################################################
#####################################   VideoClassificationTemplate   ##########################################
################################################################################################################


