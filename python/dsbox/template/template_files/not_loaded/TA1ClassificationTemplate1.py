from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class TA1ClassificationTemplate1(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1_classification_template_1",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
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
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute'),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
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
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "cast_step",
                    "primitives": [
                        "d3m.primitives.data_transformation.to_numeric.DSBOX",
                        # {
                        #     "primitive": "d3m.primitives.data_transformation.cast_to_type.Common",
                        #     "hyperparameters": {"type_to_cast": ["float"]}
                        # },
                        #"d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                    ],
                    "inputs": ["encode_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "inputs": ["cast_step"]
                },
                {
                    "name": "model_step",
                    # "runtime": {
                    #     "cross_validation": 5,
                    #     "stratified": True
                    # },
                    "primitives": [{
                        "primitive":
                            "d3m.primitives.classification.random_forest.SKlearn",
                        "hyperparameters":
                            {
                                'max_depth': [(2), (4), (8)],  # (10), #
                                'n_estimators': [(10), (20), (30)],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                    },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }


