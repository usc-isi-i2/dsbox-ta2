from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
from common_primitives.column_parser import ColumnParserPrimitive
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
            "output": "steps.5",
        'steps': [
            {
                'name': 'steps.0',
                'primitives': [
                    {
                        'primitive': 'd3m.primitives.operator.dataset_map.DataFrameCommon',
                        'hyperparameters': {
                            'fit_primitive': ['no'],
                            'primitive': [ColumnParserPrimitive],
                            'resources': ['all'],
                        },
                    },
                ],
                'inputs': ['template_input'],
            },
            {
                'name': 'steps.1',
                'primitives': [
                    {
                        'primitive': 'd3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization',
                        'hyperparameters': {
                        },
                    },
                ],
                'inputs': ['steps.0'],
            },
            {
                'name': 'steps.2',
                'primitives': [
                    {
                        'primitive': 'd3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization',
                        'hyperparameters': {
                        },
                    },
                ],
                'inputs': ['steps.1'],
            },
            {
                'name': 'steps.3',
                'primitives': [
                    {
                        'primitive': 'd3m.primitives.data_cleaning.imputer.SKlearn',
                        'hyperparameters': {
                            'use_semantic_types': [True],
                            'strategy': ['mean'],
                        },
                    },
                ],
                'inputs': ['steps.2'],
            },
            {
                'name': 'steps.4',
                'primitives': [
                    {
                        'primitive': 'd3m.primitives.regression.xgboost_gbtree.Common',
                        'hyperparameters': {
                            'n_estimators': [90, 110],
                            'learning_rate': [0.4, 0.6],
                            'max_depth': [10, 12],
                            'gamma': [0.3, 0.5],
                            'min_child_weight': [1],
                        },
                    },
                ],
                'inputs': ['steps.3', 'steps.1'],
            },
            {
                'name': 'steps.5',
                'primitives': [
                    {
                        'primitive': 'd3m.primitives.data_transformation.construct_predictions.Common',
                        'hyperparameters': {
                        },
                    },
                ],
                'inputs': ['steps.4', 'steps.1'],
            },
        ],
     }


