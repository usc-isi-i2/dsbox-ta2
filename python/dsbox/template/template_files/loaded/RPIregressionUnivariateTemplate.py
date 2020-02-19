from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class RPIregressionUnivariateTemplate(DSBoxTemplate):
    # From primitives/v2019.6.7/Distil/d3m.primitives.data_transformation.encoder.DistilTextEncoder/0.1.0/pipelines/0ed6fbca-2afd-4ba6-87cd-a3234e9846c3.json
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "RPI_regression_Univariate_Template",
            "taskType": {TaskKeyword.REGRESSION.name},
            "taskSubtype": {TaskKeyword.UNIVARIATE.name},
            "inputType": {"table"},
            "output": "steps.13",
            "steps": [
                {
                    'name': 'steps.0',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
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
                    'name': 'steps.1',
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
                    'name': 'steps.2',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                            'hyperparameters': {
                                'semantic_types': [('https://metadata.datadrivendiscovery.org/types/Attribute',)],
                            },
                        },
                    ],
                    'inputs': ['steps.1'],
                },
                {
                    'name': 'steps.3',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                            'hyperparameters': {
                                'semantic_types': [('https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/TrueTarget')],
                            },
                        },
                    ],
                    'inputs': ['steps.1'],
                },
                {
                    'name': 'encoder_step',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.feature_selection.score_based_markov_blanket.RPI',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': [ 'steps.2', 'steps.3'],
                },
                {
                    'name': 'steps.4',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_cleaning.imputer.SKlearn',
                            'hyperparameters': {
                                "use_semantic_types":[True],
                                "return_result":["replace"],
                                "strategy":["median", "most_frequent"],
                            },
                        },
                    ],
                    'inputs': ['encoder_step'],
                },
                {
                    'name': 'steps.5',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.classification.gradient_boosting.SKlearn',
                            'hyperparameters': {
                                'n_estimators': [20, 30],
                                'learning_rate': [0.3, 0.4]
                            },
                        },
                    ],
                    'inputs': ['steps.4', 'step.3'],
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
                    'inputs': ['steps.5', 'common_profiler_step'],
                },
            ]
        }


