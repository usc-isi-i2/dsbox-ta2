from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DistilacledProblemTemplate(DSBoxTemplate):
    # From primitives/v2019.6.7/Distil/d3m.primitives.data_transformation.encoder.DistilTextEncoder/0.1.0/pipelines/0ed6fbca-2afd-4ba6-87cd-a3234e9846c3.json
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Distil_acled_problem_template",
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
                                'parse_semantic_types': [('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/FloatVector')],
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
                            'primitive': 'd3m.primitives.data_transformation.data_cleaning.DistilEnrichDates',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['steps.2'],
                },
                {
                    'name': 'steps.5',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.data_cleaning.DistilReplaceSingletons',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['steps.4'],
                },
                {
                    'name': 'steps.6',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.imputer.DistilCategoricalImputer',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['steps.5'],
                },
                {
                    'name': 'steps.7',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.encoder.DistilTextEncoder',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['steps.6', 'steps.3'],
                },
                {
                    'name': 'steps.8',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.one_hot_encoder.DistilOneHotEncoder',
                            'hyperparameters': {
                                'max_one_hot': [16],
                            },
                        },
                    ],
                    'inputs': ['steps.7'],
                },
                {
                    'name': 'steps.9',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.encoder.DistilBinaryEncoder',
                            'hyperparameters': {
                                'min_binary': [17],
                            },
                        },
                    ],
                    'inputs': ['steps.8'],
                },
                {
                    'name': 'steps.10',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_cleaning.missing_indicator.SKlearn',
                            'hyperparameters': {
                                'use_semantic_types': [True],
                                'return_result': ['append'],
                                'error_on_new': [False],
                                'error_on_no_input': [False],
                            },
                        },
                    ],
                    'inputs': ['steps.9'],
                },
                {
                    'name': 'steps.11',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_cleaning.imputer.SKlearn',
                            'hyperparameters': {
                                'use_semantic_types': [True],
                                'return_result': ['replace'],
                            },
                        },
                    ],
                    'inputs': ['steps.10'],
                },
                {
                    'name': 'steps.12',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.learner.random_forest.DistilEnsembleForest',
                            'hyperparameters': {
                                'metric': ['f1Macro'],
                            },
                        },
                    ],
                    'inputs': ['steps.11', 'steps.3'],
                },
                {
                    'name': 'steps.13',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.construct_predictions.Common',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['steps.12', 'steps.1'],
                },
            ]
        }



