from dsbox.template.template import DSBoxTemplate
from d3m.metadata.problem import TaskKeyword
from dsbox.template.template_steps import TemplateSteps
from dsbox.schema import SpecializedProblem
import typing
import numpy as np  # type: ignore
from sklearn_wrap.SKExtraTreesClassifier import Hyperparams as hyper_extra_tree, SKExtraTreesClassifier
from sklearn_wrap.SKRandomForestClassifier import Hyperparams as hyper_random_forest, SKRandomForestClassifier
from sklearn_wrap.SKGradientBoostingClassifier import Hyperparams as hyper_grandient_boost, SKGradientBoostingClassifier
from sklearn_wrap.SKAdaBoostClassifier import SKAdaBoostClassifier
from sklearn_wrap.SKBaggingClassifier import SKBaggingClassifier

class NYUTimeSeriesForcastingTemplate0(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "NYU_TimeSeries_Forcasting_template_0",
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
                {
                    "name": "column_parser_step",
                    "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                "semantic_types": [("https://metadata.datadrivendiscovery.org/types/Attribute", )]
                            }
                    }],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "imputer_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_cleaning.imputer.SKlearn",
                        "hyperparameters":
                            {
                                # "use_semantic_types": [True],
                                "return_result": ["replace"],
                                "strategy":["median", "most_frequent"]
                            }
                    }],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_preprocessing.unary_encoder.DSBOX",
                            "hyperparameters":
                            {
                            }
                        }
                    ],
                    "inputs": ["imputer_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["encoder_step"],
                },
                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.feature_selection.score_based_markov_blanket.RPI",
                        "hyperparameters": {

                        }
                    }],
                    "inputs": ["to_numeric_step", "extract_target_step"]
                },

                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.passive_aggressive.SKlearn",
                            "hyperparameters": {
                            }
                        }
                    ],
                    "inputs": ["scaler_step", "extract_target_step"]
                },
                {
                    "name":"predict_step",
                    "primitives":["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs":["model_step", "column_parser_step"]
                }
            ]
        }
