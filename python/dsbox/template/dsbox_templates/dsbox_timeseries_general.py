from d3m.metadata.problem import TaskType, TaskSubtype
from .template_steps import TemplateSteps
from dsbox.template.template import DSBoxTemplate
import numpy as np




################################################################################################################
#####################################   TimeSeriesProblemsTemplates   ##########################################
################################################################################################################


class DefaultTimeseriesCollectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_timeseries_collection_template",
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "inputType": {"timeseries", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "random_forest_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [

                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                        }],
                    "inputs": ["to_dataframe_step"]
                },
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data.ColumnParser"],
                #     "inputs": ["extract_target_step"]
                # },

                # read X value
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.dsbox.TimeseriesToList"],
                    "inputs": ["to_dataframe_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": ["d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization"],
                    "inputs": ["timeseries_to_list_step"]
                },

                {
                    "name": "random_forest_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "inputs": ["random_projection_step", "extract_target_step"]
                },
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class DefaultTimeseriesRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DefaultTimeseriesRegressionTemplate",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": {TaskType.REGRESSION.name},
            "inputType": {"timeseries", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "random_forest_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [

                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                        }],
                    "inputs": ["to_dataframe_step"]
                },

                # read X value
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.dsbox.TimeseriesToList"],
                    "inputs": ["extract_attribute_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": ["d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization"],
                    "inputs": ["timeseries_to_list_step"]
                },
                {
                    "name": "cast_1_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        },
                        {
                            "primitive": "d3m.primitives.dsbox.DoNothing",
                            "hyperparameters": {}
                        }
                    ],
                    "inputs": ["extract_target_step"]
                },
                {
                    "name": "random_forest_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestRegressor"],
                    "inputs": ["random_projection_step", "cast_1_step"]
                },
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


'''
This template never working because templates "input", "outputs" schema
'''
class UCHITimeSeriesClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "UCHI_Time_Series_Classification_Template",
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "inputType": "timeseries",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                # {
                #     "name": "denormalize_step",
                #     "primitives": ["d3m.primitives.dsbox.Denormalize"],
                #     "inputs": ["template_input"]
                # },
                # {
                #     "name": "to_dataframe_step",
                #     "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                #     "inputs": ["denormalize_step"]
                # },
                # # read Y value
                # {
                #     "name": "extract_target_step",
                #     "primitives": [{
                #         "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                #         "hyperparameters":
                #             {
                #                 'semantic_types': (
                #                 'https://metadata.datadrivendiscovery.org/types/Target',
                #                 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                #                 'use_columns': (),
                #                 'exclude_columns': ()
                #             }
                #     }],
                #     "inputs": ["to_dataframe_step"]
                # },

                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.datasmash.d3m_XG2"],
                    "inputs": ["template_input", "template_input"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7
