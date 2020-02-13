from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class ISI_SDNE_Clf(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "ISI_sdne_clf",
            "taskType": {TaskKeyword.VERTEX_CLASSIFICATION.name, TaskKeyword.COMMUNITY_DETECTION.name, TaskKeyword.LINK_PREDICTION.name, TaskKeyword.GRAPH.name}, #TaskKeyword.COLLABORATIVE_FILTERING.name,
            "taskSubtype": {"NONE", TaskKeyword.NONOVERLAPPING.name, TaskKeyword.OVERLAPPING.name, TaskKeyword.MULTICLASS.name, TaskKeyword.BINARY.name, TaskKeyword.MULTILABEL.name, TaskKeyword.MULTIVARIATE.name, TaskKeyword.UNIVARIATE.name, TaskKeyword.LINK_PREDICTION.name, TaskKeyword.VERTEX_CLASSIFICATION.name, TaskKeyword.COMMUNITY_DETECTION.name},
            #"taskSubtype": "NONE",
            #"inputType": "table",
            "inputType": {"edgeList", "graph", "table"},
            "output": "model_step",
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": [
                        "d3m.primitives.data_transformation.denormalize.Common"
                        #"d3m.primitives.data_preprocessing.largest_connected_component.JHU"
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
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
                                'use_columns': (),  #'d3mIndex'),
                                'exclude_columns': ()  #[[1]]
                            }
                    }],
                    "inputs": ["common_profiler_step"]
                    },
                    {
                    "name": "embedding_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.feature_construction.sdne.DSBOX",
                        "hyperparameters": {
                            "epochs": [20, 50, 100, 200, 499],
                            "beta":[1, 2, 4, 8, 16],
                            "alpha":[0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, .1],
                            "lr":[0.0001, 0.0005, 0.001]
                        }
                    }],
                    "inputs": ["denormalize_step"]
                },
                {
                "name": "to_numeric_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX"
                }],
                "inputs": ["embedding_step"]
                },
                *TemplateSteps.classifier_model(feature_name="to_numeric_step",
                                                target_name='extract_target_step')
                # {
                #      "name": "model_step",
                #      "primitives": [{
                #          "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                #          "hyperparameters": {
                #              # 'bootstrap': ["bootstrap", "disabled"],
                #              'max_depth': [15, 30, None],
                #              'min_samples_leaf': [1, 2, 4],
                #              'min_samples_split': [2, 5, 10],
                #              'max_features': ['auto', 'sqrt'],
                #              'n_estimators': [10, 50, 100],
                #              'add_index_columns': [True],
                #              'use_semantic_types':[False],
                #              'error_on_no_input':[True],
                #              #'exclude_input_columns': [[1]]
                #              #'exclude_output_columns': ['nodeID']
                #          }
                #      }
                    #
                #{
                #    "name": "model_step",
                #    "primitives": [
                #
                #        #"d3m.primitives.classification.gaussian_classification.JHU"
                #    ],
                #        "d3m.primitives.classification.gaussian_classification.JHU"
                #    ],
                #    "inputs": ["to_numeric_step", "extract_target_step"]
                #}
            ]
        }
