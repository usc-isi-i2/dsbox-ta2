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
from dsbox.template.template import _product_dict

class DefaultSemisupervisedClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        
        graident_boost_hyper_tuning = {
                                        'max_depth': [2, 3, 4, 5],
                                        'n_estimators': [50, 60, 80, 100],
                                        'learning_rate': [0.1, 0.2, 0.4, 0.5],
                                        'min_samples_split': [2, 3],
                                        'min_samples_leaf': [1, 2],
                                      }
        extra_trees_hyper_tuning = {
                                        'bootstrap': ["bootstrap", "disabled"],
                                        'max_depth': [15, 30, None],
                                        'min_samples_leaf': [1, 2, 4],
                                        'min_samples_split': [2, 5, 10],
                                        'max_features': ['auto', 'sqrt'],
                                        'n_estimators': [10, 50, 100],
                                      }
        random_forest_hyper_tuning = {
                                        'bootstrap': ["bootstrap", "disabled"],
                                        'max_depth': [15, 30, None],
                                        'min_samples_leaf': [1, 2, 4],
                                        'min_samples_split': [2, 5, 10],
                                        'max_features': ['auto', 'sqrt'],
                                        'n_estimators': [10, 50, 100],
                                     }
        graident_boost_hyper_tuning_list = []
        hyper_gradient_boost_instance = hyper_grandient_boost.defaults()
        for hyper in _product_dict(graident_boost_hyper_tuning):
            hyper_temp = hyper_gradient_boost_instance.replace(hyper)
            graident_boost_hyper_tuning_list.append(SKGradientBoostingClassifier(hyperparams = hyper_temp))

        extra_trees_hyper_tuning_list = []
        hyper_etra_tree_instance = hyper_extra_tree.defaults()
        for hyper in _product_dict(extra_trees_hyper_tuning):
            hyper_temp = hyper_etra_tree_instance.replace(hyper)
            extra_trees_hyper_tuning_list.append(SKExtraTreesClassifier(hyperparams = hyper_temp))

        random_forest_hyper_tuning_list = []
        hyper_random_forest_instance = hyper_random_forest.defaults()
        for hyper in _product_dict(random_forest_hyper_tuning):
            hyper_temp = hyper_random_forest_instance.replace(hyper)
            random_forest_hyper_tuning_list.append(SKRandomForestClassifier(hyperparams = hyper_temp))

        all_primitives_with_different_params = []
        # all_primitives_with_different_params.extend(graident_boost_hyper_tuning_list)
        # all_primitives_with_different_params.extend(extra_trees_hyper_tuning_list)
        all_primitives_with_different_params.extend(random_forest_hyper_tuning_list)

        self.template = {
            "name": "Default_semisupervised_classification_template",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": [TaskKeyword.SEMISUPERVISED.name],
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_predictions_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() +
                     TemplateSteps.dsbox_feature_selector("classification",
                                                          first_input='data',
                                                          second_input='target') + [

                {
                    "name": "model_step", # step 6
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.semisupervised_classification.iterative_labeling.AutonBox",
                            "hyperparameters": {
                                "iters":[20],
                                "frac":[0.2, 0.1, 0.05, 0.3],
                                "blackbox": [SKRandomForestClassifier, SKGradientBoostingClassifier, SKExtraTreesClassifier, SKBaggingClassifier, SKAdaBoostClassifier],
                            }
                        }
                    ],
                    "inputs": ["feature_selector_step", "target"]
                },
                {
                    "name": "construct_predictions_step",#step 7
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "to_dataframe_step"]
                },
            ]
        }

################################################################################################
# semisupervised classification template
################################################################################################
