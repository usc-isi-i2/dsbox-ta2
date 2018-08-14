from d3m.metadata.problem import TaskType, TaskSubtype
from .template_steps import TemplateSteps
from dsbox.template.template import DSBoxTemplate
import numpy as np


################################################################################################################
#####################################   TextProblemsTemplates   ################################################
################################################################################################################


class DefaultTextClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_text_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": {"text", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_text_steps() + [
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
                                    'max_depth': [2, 5],
                                    'n_estimators': [50, 100],
                                    'learning_rate': [0.1, 0.3],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKMultinomialNB",
                            "hyperparameters":
                                {
                                    'alpha': [0, .5, 1],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                            "hyperparameters":
                                {
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


class DefaultTextRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_text_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": {"text", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_text_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKGradientBoostingRegressor",
                            "hyperparameters":
                                {
                                    'max_depth': [2, 5],
                                    'n_estimators': [100, 200],
                                    'learning_rate': [0.1, 0.3],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKRandomForestRegressor",
                            "hyperparameters":
                                {
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
