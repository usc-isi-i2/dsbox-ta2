from d3m.metadata.problem import TaskType, TaskSubtype
from .template_steps import TemplateSteps
from dsbox.template.template import DSBoxTemplate
import numpy as np


################################################################################################################
#####################################   TA1Template   ##########################################################
################################################################################################################

class TA1ClassificationTemplate1(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1_classification_template_1",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
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
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encode_step"]
                },
                {
                    "name": "cast_1_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        },
                        "d3m.primitives.dsbox.DoNothing",
                    ],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [{
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        "hyperparameters":
                            {
                                'max_depth': [(2), (4), (8)],  # (10), #
                                'n_estimators': [(10), (20), (30)]
                            }
                    },
                    ],
                    "inputs": ["cast_1_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class TA1Classification_2(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1Classification_2",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "text",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                *default_dataparser(attribute_name="extract_attribute_step",
                                    target_name="extract_target_step"),
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.dsbox.CorexText"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Labler"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "nothing_step",
                    "primitives": ["d3m.primitives.dsbox.DoNothing"],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives": ["d3m.primitives.dsbox.IQRScaler"],
                    "inputs": ["nothing_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "inputs": ["scaler_step", "extract_target_step"]
                }
            ]
        }
        # import pprint
        # pprint.pprint(self.template)
        # exit(1)

    # @override
    def importance(datset, problem_description):
        return 7


class TA1Classification_3(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1Classification_3",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
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
                    "name": "profile_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": [
                        "d3m.primitives.dsbox.CleaningFeaturizer",
                        "d3m.primitives.dsbox.DoNothing",
                    ],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "label_step",
                    "primitives": ["d3m.primitives.dsbox.Labler"],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.dsbox.CorexText"],
                    "inputs": ["label_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "model_step",
                    "primitives": [
                        "d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["corex_step", "extract_target_step"]
                }
            ]
        }

    # @override

    def importance(datset, problem_description):
        return 7


class MuxinTA1ClassificationTemplate1(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MuxinTA1ClassificationTemplate1",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
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
                    "name": "encode1_step",
                    # "primitives": ["d3m.primitives.dsbox.UnaryEncoder","d3m.primitives.dsbox.DoNothing"],
                    "primitives": ["d3m.primitives.dsbox.UnaryEncoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder", "d3m.primitives.dsbox.DoNothing"],
                    # "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["encode1_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encode2_step"]
                },
                {
                    "name": "cast_1_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        },
                        "d3m.primitives.dsbox.DoNothing",
                    ],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 2,
                        # "test_validation":1,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                            "hyperparameters":
                                {
                                    'max_depth': [(2), (4)],  # (10), #
                                    'n_estimators': [(10), (30)]
                                }
                        },
                        # {
                        #     "primitive":
                        #         "d3m.primitives.sklearn_wrap.SKLinearSVC",
                        #     "hyperparameters":
                        #         {
                        #             'C': [(1), (10)],  # (10), #
                        #         }
                        # },
                        # {
                        #     "primitive":
                        #         "d3m.primitives.sklearn_wrap.SKMultinomialNB",
                        #     "hyperparameters":
                        #         {
                        #             'alpha': [(1)],
                        #         }
                        # },
                    ],
                    "inputs": ["cast_1_step", "extract_target_step"]
                }
            ]
        }

    # @override

    def importance(datset, problem_description):
        return 7


class MuxinTA1ClassificationTemplate2(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MuxinTA1ClassificationTemplate2",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
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
                    "name": "no_op_step",
                    "primitives": ["d3m.primitives.dsbox.DoNothing"],
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
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.GreedyImputation"],
                    "inputs": ["encode_step", "extract_target_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "cast_1_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        },
                        {
                            "primitive": "d3m.primitives.dsbox.DoNothing",
                            "hyperparameters": {}
                        }
                    ],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["cast_1_step", "extract_target_step"]
                }
            ]
        }

    # @override

    def importance(datset, problem_description):
        return 7


class MuxinTA1ClassificationTemplate3(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MuxinTA1ClassificationTemplate3",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
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
                    "name": "no_op_step",
                    "primitives": ["d3m.primitives.dsbox.DoNothing"],
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
                    "inputs": ["no_op_step"]
                },
                {
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.IterativeRegressionImputation"],
                    "inputs": ["encode_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "cast_1_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        },
                        {
                            "primitive": "d3m.primitives.dsbox.DoNothing",
                            "hyperparameters": {}
                        }
                    ],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["cast_1_step", "extract_target_step"]
                }
            ]
        }

    # @override

    def importance(datset, problem_description):
        return 7


class MuxinTA1ClassificationTemplate4(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MuxinTA1ClassificationTemplate4",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
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
                    "name": "encode1_step",
                    "primitives": ["d3m.primitives.dsbox.UnaryEncoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["encode1_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encode2_step"]
                },
                {
                    "name": "cast_1_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        },
                        "d3m.primitives.dsbox.DoNothing",
                    ],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        "d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["cast_1_step", "extract_target_step"]
                }
            ]
        }

    # @override

    def importance(datset, problem_description):
        return 7


class UU3TestTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "UU3_Test_Template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            "inputType": "table",
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "multi_table_processing_step",
                    "primitives": ["d3m.primitives.dsbox.MultiTableFeaturization"],
                    "inputs": ["template_input"]
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
                    "inputs": ["multi_table_processing_step"]
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
                    "inputs": ["multi_table_processing_step"]
                },
                {
                    "name": "encode1_step",
                    "primitives": ["d3m.primitives.dsbox.UnaryEncoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["encode1_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encode2_step"]
                },
                {
                    "name": "cast_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        },
                        {
                            "primitive": "d3m.primitives.dsbox.DoNothing",
                            "hyperparameters": {}
                        }
                    ],
                    "inputs": ["impute_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestRegressor",
                        "hyperparameters":
                            {
                            }
                    }
                    ],
                    "inputs": ["cast_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7

