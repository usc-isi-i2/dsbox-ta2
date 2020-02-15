from dsbox.template.template import DSBoxTemplate 
class metalearning_binary_classification_text_template_11(DSBoxTemplate): 
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template =  {
        "weight": 20, 
        'name': 'metalearning_binary_classification_text_template_11', 
        'taskType': {'CLASSIFICATION', 'BINARY', 'TEXT'}, 
        'taskSubtype': {'CLASSIFICATION', 'BINARY', 'TEXT'}, 
        'inputType': {'text', 'table'}, 
    'output': 'steps.11',
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
                        'semantic_types': [['https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget']],
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
                        'return_result': ['replace'],
                    },
                },
            ],
            'inputs': ['steps.2'],
        },
        {
            'name': 'steps.5',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
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
                    'primitive': 'd3m.primitives.data_cleaning.imputer.SKlearn',
                    'hyperparameters': {
                        'error_on_no_input': [False],
                        'return_result': ['replace'],
                        'use_semantic_types': [True],
                        'strategy': ['mean'],
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
                    },
                },
            ],
            'inputs': ['steps.7'],
        },
        {
            'name': 'steps.9',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_preprocessing.robust_scaler.SKlearn',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.8'],
        },
        {
            'name': 'steps.10',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.classification.xgboost_gbtree.Common',
                    'hyperparameters': {
                        'learning_rate': [0.7705661042198769],
                        'add_index_columns': [True],
                        'min_child_weight': [4],
                        'n_estimators': [117],
                        'return_result': ['new'],
                        'gamma': [0.8377965803456414],
                    },
                },
            ],
            'inputs': ['steps.9', 'steps.3'],
        },
        {
            'name': 'steps.11',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_transformation.construct_predictions.Common',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.10', 'steps.2'],
        },
    ],
 }

