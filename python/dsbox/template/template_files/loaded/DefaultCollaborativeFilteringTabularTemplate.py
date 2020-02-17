from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DefaultCollaborativeFilteringTabularTemplate(DSBoxTemplate):
    '''
    Dummy implementation that does not look at the underlying graph at all.
    '''
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_Collaborative_Filtering_Tabular_Template",
            "taskType": {TaskKeyword.COLLABORATIVE_FILTERING.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype":  {TaskKeyword.COLLABORATIVE_FILTERING.name},
            "inputType": {"table"},
            "output": "construct_predictions_step",
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
                    "name": "parser_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.column_parser.Common",
                        "hyperparameters":
                            {
                                'parse_semantic_types': ([
                                    "http://schema.org/Boolean",
                                    "http://schema.org/Integer",
                                    "http://schema.org/Float",
                                    "https://metadata.datadrivendiscovery.org/types/FloatVector"
                                  ],),
                            }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (['https://metadata.datadrivendiscovery.org/types/Attribute'],),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["parser_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["extract_attribute_step"],
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["parser_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.collaborative_filtering.link_prediction.DistilCollaborativeFiltering",
                        "hyperparameters": {}
                    }
                    ],
                    "inputs": ["to_numeric_step", "extract_target_step"]
                }
                {
                    "name": "construct_predictions_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.construct_predictions.Common",
                        "hyperparameters": {}
                    }
                    ],
                    "inputs": ["model_step", "common_profiler_step"]
                }
            ]
        }

