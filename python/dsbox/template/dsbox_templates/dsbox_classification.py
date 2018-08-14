from d3m.metadata.problem import TaskType, TaskSubtype
from .template_steps import TemplateSteps
from dsbox.template.template import DSBoxTemplate
import numpy as np


################################################################################################################
##################################### General Classification Templates #########################################
################################################################################################################

################################################################################################
# A classification template encompassing several algorithms
################################################################################################


class DefaultClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKExtraTreesClassifier",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier",
                            "hyperparameters":
                                {
                                    'max_depth': [2, 3, 4, 5],
                                    'n_estimators': [50, 60, 80, 100],
                                    'learning_rate': [0.1, 0.2, 0.4, 0.5],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


# a template encompassing several NB methods
class NaiveBayesClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "naive_bayes_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKBernoulliNB",
                            "hyperparameters":
                                {
                                    'alpha': [0, .5, 1],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKGaussianNB",
                            "hyperparameters":
                                {
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKMultinomialNB",
                            "hyperparameters":
                                {
                                    'alpha': [0, .5, 1]
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class RandomForestClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "random_forest_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100]
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class ExtraTreesClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "extra_trees_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKExtraTreesClassifier",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100]
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class GradientBoostingClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "gradient_boosting_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier",
                            "hyperparameters":
                                {
                                    'max_depth': [2, 3, 5],
                                    'n_estimators': [50, 75, 100],
                                    'learning_rate': [0.1, 0.3, 0.5],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class SVCClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "svc_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKSVC",
                            "hyperparameters":
                                {
                                    'C': [0.8, 1.0, 1.2],
                                    'kernel': ['rbf', 'poly'],
                                    'degree': [2, 3, 4],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class ClassificationWithSelection(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "classification_with_feature_selection",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.human_steps() + TemplateSteps.dsbox_feature_selector("classification") +
                [
                    {
                        "name":"model_step",
                        "primitives":[
                            {
                                "primitive":"d3m.primitives.sklearn_wrap.SKSGDClassifier",
                                "hyperparameters":{
                                    "loss": ['log', 'hinge', 'squared_hinge', 'perceptron'],
                                    "alpha": [float(x) for x in np.logspace(-6, -1.004, 7)],
                                    "l1_ratio": [float(x) for x in np.logspace(-9, -0.004, 7)],
                                    "penalty": ['elasticnet', 'l2']
                                }
                            },
                            {
                                "primitive":
                                    "d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier",
                                "hyperparameters":
                                    {
                                        'max_depth': [2, 5],
                                        'n_estimators': [50, 100],
                                        'learning_rate': [0.1, 0.3],
                                        'min_samples_split': [2, 3],
                                        'min_samples_leaf': [1, 2],
                                    }
                            },

                        ],
                        "inputs":["feature_selector_step","extract_target_step"]
                    }
                ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class dsboxClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "dsbox_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING',
            # 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING',
            # 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING',
            # 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                *TemplateSteps.dsbox_preprocessing(
                    clean_name="clean_step",
                    target_name="extract_target_step"
                ),
                *TemplateSteps.dsbox_encoding(clean_name="clean_step",
                                encoded_name="encoder_step"),

                *TemplateSteps.dsbox_imputer(encoded_name="encoder_step",
                               impute_name="impute_step"),

                # *dimensionality_reduction(feature_name="impute_step",
                #                           dim_reduce_name="dim_reduce_step"),



                *TemplateSteps.classifier_model(feature_name="impute_step",
                                  target_name='extract_target_step'),
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7