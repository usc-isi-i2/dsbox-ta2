from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class RPImulticlassClassificationTabularTemplate(DSBoxTemplate):
    # From primitives/v2019.6.7/Distil/d3m.primitives.data_transformation.encoder.DistilTextEncoder/0.1.0/pipelines/0ed6fbca-2afd-4ba6-87cd-a3234e9846c3.json
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "RPI_multiclass_Classification_Tabular_Template",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.MULTICLASS.name},
            "inputType": {"table"},
            "output": "steps.8",
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
                    'primitive': 'd3m.primitives.data_transformation.remove_semantic_types.Common',
                    'hyperparameters': {
                        'columns': [(2,)],
                        'semantic_types': [('https://metadata.datadrivendiscovery.org/types/Attribute',)],
                    },
                },
            ],
            'inputs': ['common_profiler_step'],
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
            'inputs': ['steps.1'],
        },
        {
            'name': 'steps.3',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                    'hyperparameters': {
                        'semantic_types': [('https://metadata.datadrivendiscovery.org/types/Attribute',)],
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
                        'semantic_types': [('https://metadata.datadrivendiscovery.org/types/TrueTarget',)],
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
                        'method': ['fullBayesian'],
                        'nbins': [10],
                    },
                },
            ],
            'inputs': ['steps.3', 'steps.4'],
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
                        'criterion': ['entropy'],
                        'n_estimators': [32],
                    },
                },
            ],
            'inputs': ['steps.6', 'steps.4'],
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
            'inputs': ['steps.7', 'common_profiler_step'],
        },
    ],
 }