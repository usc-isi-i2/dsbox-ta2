from dsbox.template.template import DSBoxTemplate 
class metalearning_binary_classification_text_template_23(DSBoxTemplate): 
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template =  {
        "weight": 20, 
        'name': 'metalearning_binary_classification_text_template_23', 
        'taskType': {'CLASSIFICATION', 'BINARY', 'TEXT'}, 
        'taskSubtype': {'CLASSIFICATION', 'BINARY', 'TEXT'}, 
        'inputType': {'text', 'table'}, 
    'output': 'steps.9',
    'steps': [
        {
            'name': 'steps.0',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.denormalize.Common',
                    'hyperparameters': {
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
                    'primitive': 'd3m.primitives.data_preprocessing.text_reader.Common',
                    'hyperparameters': {
                        'return_result': ['new'],
                    },
                },
            ],
            'inputs': ['steps.3'],
        },
        {
            'name': 'steps.5',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_preprocessing.tfidf_vectorizer.SKlearn',
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
                    'primitive': 'd3m.primitives.data_preprocessing.standard_scaler.SKlearn',
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
                    'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                    'hyperparameters': {
                        'semantic_types': [['https://metadata.datadrivendiscovery.org/types/TrueTarget']],
                    },
                },
            ],
            'inputs': ['steps.2'],
        },
        {
            'name': 'steps.8',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.classification.linear_svc.SKlearn',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.6', 'steps.7'],
        },
        {
            'name': 'steps.9',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.construct_predictions.Common',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.8', 'steps.2'],
        },
    ],
 }

