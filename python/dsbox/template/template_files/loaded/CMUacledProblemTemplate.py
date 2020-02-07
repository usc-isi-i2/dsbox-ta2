from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class CMUacledProblemTemplate(DSBoxTemplate):
    # From primitives/v2019.6.7/Distil/d3m.primitives.data_transformation.encoder.DistilTextEncoder/0.1.0/pipelines/0ed6fbca-2afd-4ba6-87cd-a3234e9846c3.json
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "CMU_acled_problem_template",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
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
                    'name': 'steps.4',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.natural_language_processing.lda.Fastlvm',
                            'hyperparameters': {
                                "k":[10, 100, 1000, 5000],
                                "iters":[100, 1000, 5000],
                                "frac":[0.001, 0.01],
                            },
                        },
                    ],
                    'inputs': ['steps.2'],
                },
                {
                    'name': 'steps.5',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.classification.gradient_boosting.SKlearn',
                            'hyperparameters': {
                            },
                        },
                        {
                            "primitive":
                                "d3m.primitives.classification.extra_trees.SKlearn",
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
                                    'n_estimators': [10, 50, 100]
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.classification.xgboost_gbtree.Common",
                            "hyperparameters":
                                {
                                    # 'use_semantic_types': [True],
                                    # 'return_result': ['new'],
                                    'learning_rate': [0.001, 0.1],
                                    'max_depth': [15, 30, None],
                                    # 'min_samples_leaf': [1, 2, 4],
                                    # 'min_samples_split': [2, 5, 10],
                                    'n_more_estimators': [10, 50, 100, 1000],
                                    'n_estimators': [10, 50, 100, 1000]
                                }
                        }
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


