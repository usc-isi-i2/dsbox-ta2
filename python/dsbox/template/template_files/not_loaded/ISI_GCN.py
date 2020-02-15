from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class ISI_GCN(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "ISI_gcn",
            "taskType": {TaskKeyword.COLLABORATIVE_FILTERING.name, TaskKeyword.VERTEX_CLASSIFICATION.name, TaskKeyword.COMMUNITY_DETECTION.name, TaskKeyword.GRAPH.name},
            #TaskKeyword.LINK_PREDICTION.name},
            "taskSubtype":  {"NONE", TaskKeyword.NONOVERLAPPING.name, TaskKeyword.OVERLAPPING.name, TaskKeyword.MULTICLASS.name, TaskKeyword.BINARY.name, TaskKeyword.MULTILABEL.name, TaskKeyword.MULTIVARIATE.name, TaskKeyword.UNIVARIATE.name, TaskKeyword.COLLABORATIVE_FILTERING.name, TaskKeyword.VERTEX_CLASSIFICATION.name, TaskKeyword.COMMUNITY_DETECTION.name},
            "inputType": {"edgeList", "graph", "table"},
            "output": "model_step",
            "steps": [
                {
                    "name": "readgraph_step",
                    "primitives": [
                        "d3m.primitives.data_transformation.denormalize.Common"
                        #"d3m.primitives.data_preprocessing.largest_connected_component.JHU"
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["readgraph_step"]
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
                            'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                    }],
                    "inputs": ["common_profiler_step"]  #_learning"]
                },
                {
                    "name": "embedding_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.feature_construction.gcn_mixhop.DSBOX",
                        "hyperparameters":
                        {
                            'epochs': [100, 200, 300],
                            'adjacency_order':[3]
                        }
                    }],
                    "inputs": ["readgraph_step", "extract_target_step"]
                },
                *TemplateSteps.classifier_model(feature_name="embedding_step",
                                                target_name='extract_target_step')
            ]
        }
