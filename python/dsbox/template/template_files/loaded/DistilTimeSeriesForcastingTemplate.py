from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class DistilTimeSeriesForcastingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Distil_TimeSeries_Forcasting_Template",
            "taskType": TaskKeyword.TIME_SERIES.name,
            "taskSubtype": {"FORECASTING"},
            "inputType": {"table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                        "hyperparameters": {
                            "drop_non_numeric_columns": [False]
                        }
                    }],
                    "inputs": ["pre_extract_target_step"]
                },

                # read X value
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "imputer_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_cleaning.imputer.SKlearn",
                        "hyperparameters":
                            {
                                'return_result': ['new'],
                                'use_semantic_types': [True],
                            }
                    }],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.data_transformation.grouping_field_compose.Common"],
                    "inputs": ["imputer_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.time_series_forecasting.lstm.DeepAR",
                            "hyperparameters": {
                                'count_data': [True],
                            }
                        }
                    ],
                    "inputs": ["timeseries_to_list_step", "extract_target_step"]
                },
                {
                    "name": "construct_prediction_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.Common",
                            "hyperparameters": {
                            }
                        }
                    ],
                    "inputs": ["model_step", "common_profiler_step"]
                },
            ]
        }

