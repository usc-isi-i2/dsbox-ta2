from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class BBNacledProblemTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "BBN_acled_problem_template",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "inputType": {"table"},
            "output": "prediction_step",
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
                    "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                    "inputs": ["common_profiler_step"]
                },
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
                            'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                    }],
                    "inputs": ["parser_step"]
                },
                {
                    "name": "count_vectorizer_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_preprocessing.count_vectorizer.SKlearn",
                        "hyperparameters":
                        {
                            'use_semantic_types':[True],
                            'return_result': ["replace"],
                        }
                    }],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "tfidf_vectorizer_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.feature_extraction.tfidf_vectorizer.BBN",
                        "hyperparameters":
                        {
                            'norm':["l1", "l2"],
                            'smooth_idf': [False, True],
                            'sublinear_tf': [True, False],
                            'use_idf': [False, True],
                        }
                    }],
                    "inputs": ["count_vectorizer_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.classification.mlp.SKlearn",
                        # TODO: need to add hyperparams tunning for mlp primitive here
                        "hyperparameters": {

                        }
                    }
                    ],
                    "inputs": ["tfidf_vectorizer_step", "extract_target_step"]
                },
                {
                    "name": "prediction_step",
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "common_profiler_step"]
                }

            ]
        }


