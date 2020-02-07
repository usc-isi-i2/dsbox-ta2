from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class NaiveBayesClassificationTemplate(DSBoxTemplate):
    '''A template encompassing several NB methods'''
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "naive_bayes_classification_template",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.classification.bernoulli_naive_bayes.SKlearn",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'return_result': ['new'],
                                    'add_index_columns': [True],
                                    'alpha': [0, .5, 1],
                                }
                        },
                        # {
                        #     "primitive":
                        #         "d3m.primitives.classification.gaussian_naive_bayes.SKlearn",
                        #     "hyperparameters":
                        #         {
                        #             'use_semantic_types': [True],
                        #             'return_result': ['new'],
                        #             'add_index_columns': [True],
                        #         }
                        # },
                        # {
                        #     "primitive":
                        #         "d3m.primitives.classification.multinomial_naive_bayes.SKlearn",
                        #     "hyperparameters":
                        #         {
                        #             'use_semantic_types': [True],
                        #             'return_result': ['new'],
                        #             'add_index_columns': [True],
                        #             'alpha': [0, .5, 1]
                        #         }
                        # },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


