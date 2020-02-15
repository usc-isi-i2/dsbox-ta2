from dsbox.template.template import DSBoxTemplate 
class metalearning_forecasting_tabular_timeSeries_template_36(DSBoxTemplate): 
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template =  {
        "weight": 20, 
        'name': 'metalearning_forecasting_tabular_timeSeries_template_36', 
        'taskType': {'FORECASTING', 'TIMESERIES', 'TABULAR'}, 
        'taskSubtype': {'FORECASTING', 'TIMESERIES', 'TABULAR'}, 
        'inputType': {'table'}, 
    'output': 'steps.8',
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
                    'primitive': 'd3m.primitives.data_cleaning.imputer.SKlearn',
                    'hyperparameters': {
                        'strategy': ['most_frequent'],
                    },
                },
            ],
            'inputs': ['steps.3'],
        },
        {
            'name': 'steps.5',
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
            'name': 'steps.6',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.feature_selection.score_based_markov_blanket.RPI',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.4', 'steps.5'],
        },
        {
            'name': 'steps.7',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.regression.random_forest.SKlearn',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.6', 'steps.5'],
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
            'inputs': ['steps.7', 'steps.2'],
        },
    ],
 }

