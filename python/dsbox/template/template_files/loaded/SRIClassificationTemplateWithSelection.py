from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 

# what does this template exactly used for ?????
class SRIClassificationTemplateWithSelection(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "weight": 50,
            "name": "SRI_classification_template_with_selection",
            "taskSubtype": {TaskKeyword.VERTEX_CLASSIFICATION.name},
            "taskType": {TaskKeyword.VERTEX_CLASSIFICATION.name},
            # "taskType": {TaskKeyword.VERTEX_CLASSIFICATION.name, TaskKeyword.COMMUNITY_DETECTION.name, TaskKeyword.LINK_PREDICTION.name, TaskKeyword.TIME_SERIES.name},
            # "taskSubtype": {"NONE", TaskKeyword.NONOVERLAPPING.name, TaskKeyword.OVERLAPPING.name, TaskKeyword.MULTICLASS.name, TaskKeyword.BINARY.name, TaskKeyword.MULTILABEL.name, TaskKeyword.MULTIVARIATE.name, TaskKeyword.UNIVARIATE.name, TaskKeyword.TIME_SERIES.name},
            #"inputType": "table",
            "inputType": {"edgeList", "graph", "table"},
            "output": "prediction_step",
            "steps": [
                {
                    "name": "text_reader_step",
                    "primitives": ["d3m.primitives.data_preprocessing.dataset_text_reader.DatasetTextReader"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["text_reader_step"]
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
                    "name": "parser_step",
                    "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "pre_extract_target_step",
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
                    "name": "extract_target_step",
                    "primitives": ["d3m.primitives.data_transformation.simple_column_parser.DataFrameCommon"],
                    "inputs": ["pre_extract_target_step"]
                },
                {
                    "name": "extract_attribute_step1",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                        {
                            'semantic_types': [("https://metadata.datadrivendiscovery.org/types/PrimaryKey",
                                                "https://metadata.datadrivendiscovery.org/types/GroupingKey",
                                                "https://metadata.datadrivendiscovery.org/types/UniqueKey",
                                                "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
                                                "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey",)],
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                    }],
                    "inputs": ["parser_step"]
                },
                {
                    "name": "extract_attribute_step2",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                        {
                            'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                    }],
                    "inputs": ["extract_attribute_step1"]
                },
                {
                    "name": "data_conditioner_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.conditioner.Conditioner",
                        "hyperparameters":
                        {
                            "ensure_numeric":[True, False],
                            "maximum_expansion": [30]
                        }
                    }],
                    "inputs": ["extract_attribute_step2"]
                },
                {
                    "name": "select_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.feature_selection.select_percentile.SKlearn",
                        "hyperparameters":
                        {
                            "score_func": ["f_classif"],
                            "percentile": [30, 40, 50, 60, 68],
                            "use_semantic_types": [False],
                            "add_index_columns": [False],
                            "error_on_no_input": [True],
                            "return_semantic_type": ["https://metadata.datadrivendiscovery.org/types/PredictedTarget"]
                        }
                    },
                    {
                            "primitive": "d3m.primitives.feature_selection.variance_threshold.SKlearn",
                            "hyperparameters":{
                                'threshold':[0.0],
                            }
                        }
                    ],
                    "inputs": ["data_conditioner_step", "extract_target_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive":
                            "d3m.primitives.classification.bernoulli_naive_bayes.SKlearn",
                            "hyperparameters":
                            {
                                'alpha': [0.01, 0.1, 1.0],
                                'binarize': [0.0],
                                'fit_prior': [False],
                                'return_result': ["new"],
                                'use_semantic_types': [False],
                                'add_index_columns': [False],
                                'error_on_no_input':[True],
                            }
                        }, 
                        {
                            "primitive": "d3m.primitives.regression.gradient_boosting.SKlearn",
                        },
                        {"primitive": "d3m.primitives.classification.random_forest.SKlearn"
                        }
                        ],
                    "inputs": ["select_step", "extract_target_step"]
                },
                {
                    "name": "prediction_step",
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "to_dataframe_step"]
                }
            ]
        }


