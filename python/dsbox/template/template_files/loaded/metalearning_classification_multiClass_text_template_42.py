from dsbox.template.template import DSBoxTemplate 
class metalearning_classification_multiClass_text_template_42(DSBoxTemplate): 
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template =  {
        "weight": 20, 
        'name': 'metalearning_classification_multiClass_text_template_42', 
        'taskType': {'CLASSIFICATION', 'TEXT', 'MULTICLASS'}, 
        'taskSubtype': {'CLASSIFICATION', 'TEXT', 'MULTICLASS'}, 
        'inputType': {'text', 'table'}, 
    'output': 'steps.7',
    'steps': [
        {
            'name': 'steps.0',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.denormalize.Common',
                    'hyperparameters': {
                        'starting_resource': ['learningData'],
                    },
                },
            ],
            'inputs': ['template_input'],
        },
        {
            'name': 'steps.1',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
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
                        'semantic_types': [['https://metadata.datadrivendiscovery.org/types/Attribute']],
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
                        'semantic_types': [['https://metadata.datadrivendiscovery.org/types/TrueTarget']],
                    },
                },
            ],
            'inputs': ['steps.1'],
        },
        {
            'name': 'steps.5',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_cleaning.imputer.SKlearn',
                    'hyperparameters': {
                        'error_on_no_input': [False],
                        'return_result': ['replace'],
                        'use_semantic_types': [True],
                    },
                },
            ],
            'inputs': ['steps.3'],
        },
        {
            'name': 'steps.6',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.classification.random_classifier.Test',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.5', 'steps.4'],
        },
        {
            'name': 'steps.7',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.construct_predictions.Common',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.6', 'steps.1'],
        },
    ],
 }

