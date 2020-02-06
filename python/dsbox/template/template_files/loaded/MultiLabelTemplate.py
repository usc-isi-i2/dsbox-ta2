from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class MultiLabelTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Multi_Label_tempalte",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.MULTILABEL.name},
            "inputType": {"table"},
            "output": "prediction_step",
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
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["steps.1"]
                },
                {
                    'name': 'steps.2',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                            'hyperparameters': {
                                'semantic_types': [
                                    ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                     'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
                                     'https://metadata.datadrivendiscovery.org/types/Attribute')],
                            },
                        },
                    ],
                    'inputs': ['common_profiler_step'],
                },
                {
                    'name': 'steps.3',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                            'hyperparameters': {
                                'semantic_types': [
                                    ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                     'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
                                     'https://metadata.datadrivendiscovery.org/types/SuggestedTarget')],
                            },
                        },
                    ],
                    'inputs': ['common_profiler_step'],
                },
                {
                    'name': 'steps.4',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.column_parser.Common',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['steps.2'],
                },
                {
                    'name': 'prediction_step',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.classification.multilabel_classifier.DSBOX',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['steps.4', 'steps.3'],
                },
            ],
        }

