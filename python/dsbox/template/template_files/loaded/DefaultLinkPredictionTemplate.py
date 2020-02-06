from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DefaultLinkPredictionTemplate(DSBoxTemplate):
    '''
    Dummy implementation that does not look at the underlying graph at all.
    '''
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_LinkPrediction_Template",
            "taskType": {TaskKeyword.GRAPH_MATCHING.name, TaskKeyword.GRAPH.name, TaskKeyword.LINK_PREDICTION.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype":  {"NONE", TaskKeyword.GRAPH_MATCHING.name, TaskKeyword.GRAPH.name, TaskKeyword.LINK_PREDICTION.name},
            "inputType": {"graph"},
            "output": "model_step",
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
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["common_profiler_step"]
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
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                        "hyperparameters": {
                            # 'bootstrap': ["bootstrap", "disabled"],
                            'max_depth': [15, 30, None],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10],
                            'max_features': ['auto', 'sqrt'],
                            'n_estimators': [10, 50, 100],
                            'add_index_columns': [True],
                            'use_semantic_types':[True],
                        }
                    }
                    ],
                    "inputs": ["to_numeric_step", "extract_target_step"]
                }
            ]
        }

