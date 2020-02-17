from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class CMUmultiClassificationTabularTemplate(DSBoxTemplate):
    # From primitives/v2019.6.7/Distil/d3m.primitives.data_transformation.encoder.DistilTextEncoder/0.1.0/pipelines/0ed6fbca-2afd-4ba6-87cd-a3234e9846c3.json
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "CMU_multi_Classification_Tabular_Template",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.TABULAR.name, TaskKeyword.MULTICLASS.name},
            "inputType": {"table"},
            "output": "steps.8",    
                'steps': [
        {
            'name': 'steps.1',
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
                "name": "profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                "inputs": ["steps.1"]
            },
        {
            'name': 'steps.2',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.column_parser.Common',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['profiler_step'],
        },
        {
            'name': 'steps.3',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                    'hyperparameters': {
                        'semantic_types': [('https://metadata.datadrivendiscovery.org/types/TrueTarget',)],
                    },
                },
            ],
            'inputs': ['steps.2'],
        },
        {
            'name': 'steps.4',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
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
                    'primitive': 'd3m.primitives.feature_selection.joint_mutual_information.AutoRPI',
                    'hyperparameters': {
                        'method': ['counting'],
                        'nbins': [10],
                    },
                },
            ],
            'inputs': ['steps.4', 'steps.3'],
        },
        {
            'name': 'steps.6',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_cleaning.imputer.SKlearn',
                    'hyperparameters': {
                        'strategy': ['most_frequent'],
                    },
                },
            ],
            'inputs': ['steps.5'],
        },
        {
            'name': 'steps.7',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.classification.extra_trees.SKlearn',
                    'hyperparameters': {
                        'n_estimators': [10],
                    },
                },
            ],
            'inputs': ['steps.6', 'steps.3'],
        },
        {
            'name': 'steps.8',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.construct_predictions.Common',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.7', 'steps.1'],
        },
    ],
 }