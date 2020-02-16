from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class MITregresionRelationalTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MIT_regresion_Relational_Template",
            "taskType": {TaskKeyword.REGRESSION.name},
            "taskSubtype": {TaskKeyword.RELATIONAL.name, TaskKeyword.UNIVARIATE.name},
            "inputType": {"table"},
            "output": "steps.13",
            "steps": [
                {
                    'name': 'steps.0',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.operator.dataset_map.DataFrameCommon',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['template_input'],
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["steps.0"]
                },
                {
                    'name': 'parser_step',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.column_parser.Common',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['common_profiler_step'],
                },
                {
                    'name': 'deep_feature_synthesis_step',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['parser_step'],
                },
                {
                    'name': 'model_step',
                    'primitives': [
                        {
                            "primitive":
                                "d3m.primitives.regression.xgboost_gbtree.Common",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'return_result': ['new'],
                                    'add_index_columns': [True],
                                    'bootstrap': ["bootstrap", "disabled"],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [80, 120]
                                }
                        },
                    ],
                    'inputs': ['deep_feature_synthesis_step', 'parser_step'],
                },
                {
                    'name': 'steps.6',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.construct_predictions.Common',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['model_step', 'parser_step'],
                },
            ]
        }


