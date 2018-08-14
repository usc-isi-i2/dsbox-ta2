from d3m.metadata.problem import TaskType, TaskSubtype
from .template_steps import TemplateSteps
from dsbox.template.template import DSBoxTemplate
import numpy as np



################################################################################################################
#####################################   GraphProblemsTemplates   ###############################################
################################################################################################################


class SRILinkPredictionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_LinkPrediction_Template",
            "taskType": {TaskType.LINK_PREDICTION.name,TaskType.GRAPH_MATCHING.name, TaskType.VERTEX_NOMINATION.name},
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.datasets.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name":"to_dataframe_step",
                    "primitives":["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs":["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                        }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                        }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "primitives":[{
                        "primitive":"d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        "hyperparameters":{
                                    # 'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                            }
                        }
                    ],
                    "inputs":["extract_attribute_step", "extract_target_step"]
                }
            ]
        }

    def importance(datset, problem_description):
        return 7


class SRIGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_GraphMatching_Template",
            "taskType": {TaskType.GRAPH_MATCHING.name, TaskType.LINK_PREDICTION.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive":"d3m.primitives.sri.psl.GraphMatchingLinkPrediction",
                            "hyperparameters":{
                                "link_prediction_hyperparams" : [(TemplateSteps.class_hyperparameter_generator("d3m.primitives.sri.psl.GraphMatchingLinkPrediction", "link_prediction_hyperparams", {"truth_threshold": 0.0000001,"psl_options": "","psl_temp_dir": "/tmp/psl/run","postgres_db_name": "psl_d3m","admm_iterations": 1000,"max_threads": 0,"jvm_memory": 0.75,"prediction_column": "match"}))]
                                }
                        }
                    ],
                    "inputs": ["template_input"]
                }
            ]
        }

    def importance(dataset, problem_description):
        return 7


class SRIVertexNominationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Vertex_Nomination_Template",
            "taskType": TaskType.VERTEX_NOMINATION.name,
            "taskSubtype": "NONE",
            "inputType": {"graph", "edgeList"},
            "output": "model_step",
            "steps": [
                {
                    "name": "parse_step",
                    "primitives": ["d3m.primitives.sri.graph.VertexNominationParser"],
                    "inputs": ["template_input"]

                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sri.psl.VertexNomination"],
                    "inputs": ["parse_step"]

                }
            ]
        }

    def importance(datset, problem_description):
        return 7


class SRICollaborativeFilteringTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Collaborative_Filtering_Template",
            "taskType": {TaskType.COLLABORATIVE_FILTERING.name},
            "taskSubtype": "NONE",
            "inputType": "table",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sri.psl.CollaborativeFilteringLinkPrediction"],
                    "inputs": ["template_input"]

                }
            ]
        }

    def importance(dataset, problem_description):
        return 7


class SRICommunityDetectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Community_Detection_Template",
            "taskType": {TaskType.COMMUNITY_DETECTION.name},
            "taskSubtype": {TaskSubtype.NONOVERLAPPING.name, TaskSubtype.OVERLAPPING.name},
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "parser_step",
                    "primitives": ["d3m.primitives.sri.graph.CommunityDetectionParser"],
                    "inputs": ["template_input"]

                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sri.psl.CommunityDetection"],
                    "inputs": ["parser_step"]
                }
            ]
        }

    def importance(dataset, problem_description):
        return 7


'''
JHU needs R supports and their primitives seem to be failing in pickling
'''
class JHUVertexNominationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Vertex_Nomination_Template",
            "taskType": TaskType.VERTEX_NOMINATION.name,
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.jhu_primitives.SpectralGraphClustering"],
                    "inputs": ["template_input"]

                }
            ]
        }

    def importance(datset, problem_description):
        return 7


class JHUGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Graph_Matching_Template",
            "taskType": TaskType.GRAPH_MATCHING.name,
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.jhu_primitives.SeededGraphMatching"],
                    "inputs": ["template_input"]
                }
            ]
        }

    def importance(datset, problem_description):
        return 7

