from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class ARIMATemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "ARIMA_Template",
            "taskType": TaskKeyword.TIME_SERIES.name,
            "taskSubtype": {"FORECASTING"},
            "inputType": {"table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "ARIMA_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "parser_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.column_parser.Common",
                        "hyperparameters": {
                            "parse_semantic_types": [('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/FloatVector'),]
                        }
                    }],
                    "inputs": ["common_profiler_step"]
                },

                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                             'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                            }
                    }],
                    "inputs": ["parser_step"]
                },

                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["parser_step"]
                },
                # {
                #     "name": "extract_attribute_step",
                #     "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                #     "inputs": ["pre_extract_attribute_step"]
                # },
                {
                    "name": "ARIMA_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.time_series_forecasting.arima.DSBOX",
                            "hyperparameters": {
                                "take_log": [(False)],
                                "auto": [(True)]
                            }
                        },
                        {
                            "primitive": "d3m.primitives.time_series_forecasting.rnn_time_series.DSBOX",
                            "hyperparameters": {
                            }
                        }
                    ], # can add tuning parameters like: auto-fit, take_log, etc
                    "inputs": ["extract_attribute_step", "extract_target_step"]
                },
            ]
        }

