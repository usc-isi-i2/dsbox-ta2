from dsbox.template.template import DSBoxTemplate 
class metalearning_classification_multiClass_timeSeries_template_273(DSBoxTemplate): 
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template =  {
        "weight": 20, 
        'name': 'metalearning_classification_multiClass_timeSeries_template_273', 
        'taskType': {'CLASSIFICATION', 'TIMESERIES', 'MULTICLASS'}, 
        'taskSubtype': {'CLASSIFICATION', 'TIMESERIES', 'MULTICLASS'}, 
        'inputType': {'timeseries', 'table'}, 
    'output': 'steps.7',
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
                    'primitive': 'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX',
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
                    'primitive': 'd3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.2'],
        },
        {
            'name': 'steps.4',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.data_preprocessing.robust_scaler.SKlearn',
                    'hyperparameters': {
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
            'inputs': ['steps.1'],
        },
        {
            'name': 'steps.6',
            'primitives': [
                {
                    'primitive': 'd3m.primitives.classification.logistic_regression.SKlearn',
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
                    'primitive': 'd3m.primitives.data_transformation.construct_predictions.Common',
                    'hyperparameters': {
                    },
                },
            ],
            'inputs': ['steps.6', 'steps.1'],
        },
    ],
 }

