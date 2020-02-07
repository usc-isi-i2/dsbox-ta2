from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class TimeseriesLstmFcnTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TimeseriesLstmFcnTemplate",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": {"timeseries", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "lstm_fcn_step",  # Name of the final step generating the prediction
            "target": "steps.4",  # Name of the step generating the ground truth
            'output': 'steps.6',
            'steps': [
                {
                    'name': 'steps.0',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter',
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
                            'primitive': 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                            'hyperparameters': {
                            },
                        },
                    ],
                    'inputs': ['template_input'],
                },
                {
                    'name': 'steps.3',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.data_transformation.column_parser.Common',
                            'hyperparameters': {
                                'parse_semantic_types': [('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/FloatVector')],
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
                                'semantic_types': [('https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget')],
                            },
                        },
                    ],
                    'inputs': ['steps.3'],
                },
                {
                    'name': 'steps.5',
                    'primitives': [
                        {
                            'primitive': 'd3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN',
                            'hyperparameters': {
                                'use_multiprocessing': [False],
                                'attention_lstm': [True],
                            },
                        },
                    ],
                    'inputs': ['steps.1', 'steps.4'],
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
                    'inputs': ['steps.5', 'steps.2'],
                },
            ]
            # "steps": [
            #     # {
            #     #     "name": "common_profiler_step",
            #     #     "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
            #     #     "inputs": ["template_input"]
            #     # },
            #     # {
            #     #     "name": "timeseries_format_step",
            #     #     "primitives": ["d3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter"],
            #     #     "inputs": ["common_profiler_step"]
            #     # },
            #     {
            #         "name": "timeseries_format_step",
            #         "primitives": ["d3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter"],
            #         "inputs": ["template_input"]
            #     },
            #     {
            #         "name": "to_dataframe_step",
            #         "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
            #         "inputs": ["timeseries_format_step"]
            #     },
            #     {
            #         "name": "to_dataframe_step_2",
            #         "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
            #         "inputs": ["template_input"]
            #     },
            #     {
            #         "name": "parser_step",
            #         "primitives": [
            #             {
            #                 "primitive": "d3m.primitives.data_transformation.column_parser.Common",
            #                 "hyperparameters": {
            #                     "parse_semantic_types": [(
            #                         "http://schema.org/Boolean",
            #                         "http://schema.org/Integer",
            #                         "http://schema.org/Float",
            #                         "https://metadata.datadrivendiscovery.org/types/FloatVector")],
            #                 }
            #             }
            #         ],
            #         "inputs": ["to_dataframe_step"],
            #     },
            #     {
            #         "name": "extract_target_step",
            #         "primitives": [
            #             {
            #                 "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
            #                 "hyperparameters": {
            #                     "semantic_types": [
            #                         ("https://metadata.datadrivendiscovery.org/types/Target",
            #                          "https://metadata.datadrivendiscovery.org/types/TrueTarget",
            #                          "https://metadata.datadrivendiscovery.org/types/SuggestedTarget")
            #                     ]
            #                 }
            #             }
            #         ],
            #         "inputs": ["parser_step"]
            #     },
            #     {
            #         "name": "lstm_fcn_step",
            #         "primitives": ["d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN"],
            #         "inputs": ["to_dataframe_step", "extract_target_step"]
            #     },
            #     {
            #         "name": "",
            #         "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
            #         "inputs": ["lstm_fcn_step", "to_dataframe_step_2"]
            #     }
            # ]
        }


