import copy
import logging
import numpy as np  # type: ignore
import typing

import d3m
from d3m import index

from d3m.metadata.problem import TaskType, TaskSubtype
from dsbox.schema import SpecializedProblem
from dsbox.template.template import DSBoxTemplate
from .template_steps import TemplateSteps

_logger = logging.getLogger(__name__)

if d3m.__version__ == "2019.4.4":
    from .utils import SEMANTIC_TYPES
else:
    from d3m.container.dataset import D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES, D3M_RESOURCE_TYPE_CONSTANTS_TO_SEMANTIC_TYPES, D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES
# for d3m V2019.5.8: new SEMANTIC_TYPES need to be finished like this
    SEMANTIC_TYPES = D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES.copy()
    SEMANTIC_TYPES.update(D3M_RESOURCE_TYPE_CONSTANTS_TO_SEMANTIC_TYPES)
    SEMANTIC_TYPES.update(D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES)


class TemplateLibrary:
    """
    Library of template pipelines
    """

    def __init__(self, library_dir: str = None, run_single_template: str = "") -> None:
        self.templates: typing.List[typing.Type[DSBoxTemplate]] = []
        self.primitive: typing.Sequence[str] = index.search()

        self.library_dir = library_dir
        if self.library_dir is None:
            self._load_library()

        self.all_templates = {
            "test_default_classification_template":TestDefaultClassificationTemplate,
            "data_augment_regression_template":DataAugmentRegressionTemplate,
            "Horizontal_Template": HorizontalTemplate,
            "default_classification_template": DefaultClassificationTemplate,
            "default_regression_template": DefaultRegressionTemplate,
            "classification_with_feature_selection": ClassificationWithSelection,
            "regression_with_feature_selection": RegressionWithSelection,

            "Large_column_number_with_numerical_only_classification": Large_column_number_with_numerical_only_classification,
            "Large_column_number_with_numerical_only_regression": Large_column_number_with_numerical_only_regression,

            # new classification
            "random_forest_classification_template": RandomForestClassificationTemplate,
            "extra_trees_classification_template": ExtraTreesClassificationTemplate,
            "gradient_boosting_classification_template": GradientBoostingClassificationTemplate,
            "svc_classification_template": SVCClassificationTemplate,
            "naive_bayes_classification_template": NaiveBayesClassificationTemplate,

            # new regression
            "random_forest_regression_template": RandomForestRegressionTemplate,
            "extra_trees_regression_template": ExtraTreesRegressionTemplate,
            "gradient_boosting_regression_template": GradientBoostingRegressionTemplate,
            "svr_regression_template": SVRRegressionTemplate,
            "Testing_template": TESTINGTemplate, # template from elastic search
            "Alpha_Zero_template": AlphaZeroEvalTemplate, # template from elastic search
            # older templates
            "dsbox_classification_template": dsboxClassificationTemplate,
            "dsbox_regression_template": dsboxRegressionTemplate,
            "CMU_Clustering_Template": CMUClusteringTemplate,
            #"Default_timeseries_collection_template": DefaultTimeseriesCollectionTemplate,
            "Default_image_processing_regression_template":
                DefaultImageProcessingRegressionTemplate,
            "TA1_classification_template_1": TA1ClassificationTemplate1,
            "MuxinTA1ClassificationTemplate1": MuxinTA1ClassificationTemplate1,
            "MuxinTA1ClassificationTemplate2": MuxinTA1ClassificationTemplate2,
            "MuxinTA1ClassificationTemplate3": MuxinTA1ClassificationTemplate3,
            "UU3_Test_Template": UU3TestTemplate,
            "TA1Classification_2": TA1Classification_2,
            "TA1Classification_3": TA1Classification_3,
            "TA1VggImageProcessingRegressionTemplate": TA1VggImageProcessingRegressionTemplate,
            "Default_LinkPrediction_Template": DefaultLinkPredictionTemplate,

            # graph
            "ISI_graph_norm_clf": ISIGraphNormClf,
            "ISI_gcn": ISI_GCN,

            # text
            "default_text_classification_template": DefaultTextClassificationTemplate,
            "default_text_regression_template": DefaultTextRegressionTemplate,

            "BBN_audio_classification_template": BBNAudioClassificationTemplate,
            "SRI_GraphMatching_Template": SRIGraphMatchingTemplate,
            "SRI_Vertex_Nomination_Template": SRIVertexNominationTemplate,
            "SRI_Collaborative_Filtering_Template": SRICollaborativeFilteringTemplate,
            "SRI_Community_Detection_Template": SRICommunityDetectionTemplate,
            "SRI_Mean_BaseLine_Template": SRIMeanBaselineTemplate,
            "UCHI_Time_Series_Classification_Template": UCHITimeSeriesClassificationTemplate,
            "JHU_Graph_Matching_Template": JHUGraphMatchingTemplate,
            "JHU_Vertex_Nomination_Template": JHUVertexNominationTemplate,
            "Default_Time_Series_Forcasting_Template": DefaultTimeSeriesForcastingTemplate,
            "Default_timeseries_collection_template": DefaultTimeseriesCollectionTemplate,
            "Michigan_Video_Classification_Template": MichiganVideoClassificationTemplate,
            "DefaultTimeseriesRegressionTemplate": DefaultTimeseriesRegressionTemplate,
            # "DefaultImageClassificationWithCNNTemplate": DefaultImageClassificationWithCNNTemplate,
            "ARIMA_Template": ARIMATemplate,
            "TimeSeriesForcastingTestingTemplate": TimeSeriesForcastingTestingTemplate,
            "DefaultObjectDetectionTemplate": DefaultObjectDetectionTemplate,
            "DefaultVideoClassificationTemplate": DefaultVideoClassificationTemplate,

            # Specialized problems: privileged data
            "LupiPriviledgedInformationClassification": LupiPriviledgedInformationClassification,
        }

        if run_single_template:
            self._load_single_inline_templates(run_single_template)
        else:
            self._load_inline_templates()

    def get_templates(self, task: TaskType, subtype: TaskSubtype, taskSourceType: typing.Set,
                      specialized_problem: SpecializedProblem = SpecializedProblem.NONE) -> typing.List[DSBoxTemplate]:
        results: typing.List[DSBoxTemplate] = []
        results.append(SRIMeanBaselineTemplate())  # put the meanbaseline here so whatever dataset will have a result
        for template_class in self.templates:
            template = template_class()
            # sourceType refer to d3m/container/dataset.py ("SEMANTIC_TYPES" as line 40-70)
            # taskType and taskSubtype refer to d3m/
            if task.name in template.template['taskType'] and subtype.name in template.template['taskSubtype']:
                # if there is only one task source type which is table, we don't need to check
                # other things
                taskSourceType_check = copy.copy(taskSourceType)
                if {"table"} == taskSourceType_check and "table" in template.template['inputType']:
                    results.append(template)
                else:
                    # otherwise, we need to process in another way because "table" source type
                    # exist nearly in every dataset
                    if "table" in taskSourceType_check:
                        taskSourceType_check.remove("table")

                    for each_source_type in taskSourceType_check:
                        if type(template.template['inputType']) is set:
                            if each_source_type in template.template['inputType']:
                                results.append(template)
                        else:
                            if each_source_type == template.template['inputType']:
                                results.append(template)

        if not specialized_problem == SpecializedProblem.NONE:
            _logger.debug(f'Specialized problem: {specialized_problem}')
            results = [template for template in results
                       if 'specializedProblem' in template.template
                       and specialized_problem in template.template['specializedProblem']]
        # if we finally did not find a proper template to use
        if results == []:
            _logger.error(f"Cannot find a suitable template type to fit the problem: {task.name}")
        else:
            # otherwise print the template list we added
            for each_template in results:
                _logger.info(f"{each_template} has been added to template base.")

        return results

    def _load_library(self):
        # TODO
        # os.path.join(library_dir, 'template_library.yaml')
        pass

    def _load_inline_templates(self):
        # pass # if no is loading
        # template that gives us the mean baseline as a result
        # self.templates.append(SRIMeanBaselineTemplate)
        # horzontalTemplate
        # self.templates.append(HorizontalTemplate)
        # self.templates.append(DataAugmentRegressionTemplate)

        # default tabular templates, encompassing many of the templates below
        self.templates.append(DefaultClassificationTemplate)
        self.templates.append(NaiveBayesClassificationTemplate)
        self.templates.append(DefaultRegressionTemplate)

        # new tabular classification
        # Muxin said it was already included in DefaultClassification
        # self.templates.append(RandomForestClassificationTemplate)
        # self.templates.append(ExtraTreesClassificationTemplate)
        # self.templates.append(GradientBoostingClassificationTemplate)

        # takes too long to run
        self.templates.append(SVCClassificationTemplate)

        # new tabular regression
        self.templates.append(RandomForestRegressionTemplate)
        self.templates.append(ExtraTreesRegressionTemplate)
        self.templates.append(GradientBoostingRegressionTemplate)
        self.templates.append(AlphaZeroEvalTemplate)
        self.templates.append(TESTINGTemplate)
        # takes too long to run
        # self.templates.append(SVRRegressionTemplate)

        # text templates, but also support tabular data
        self.templates.append(DefaultTextClassificationTemplate)
        self.templates.append(DefaultTextRegressionTemplate)

        # Tabular Classification
        self.templates.append(TA1Classification_3)
        self.templates.append(UU3TestTemplate)

        # Image related
        self.templates.append(DefaultImageProcessingRegressionTemplate)
        self.templates.append(DefaultObjectDetectionTemplate)
        self.templates.append(DefaultVideoClassificationTemplate)
        # self.templates.append(DefaultImageClassificationWithCNNTemplate)
        self.templates.append(TA1VggImageProcessingRegressionTemplate)

        # Privileged information
        self.templates.append(LupiPriviledgedInformationClassification)

        # Others
        self.templates.append(DefaultTimeseriesCollectionTemplate)
        self.templates.append(TimeSeriesForcastingTestingTemplate)
        self.templates.append(ARIMATemplate)
        self.templates.append(DefaultTimeseriesRegressionTemplate)

        self.templates.append(DefaultLinkPredictionTemplate)
        self.templates.append(SRICommunityDetectionTemplate)
        self.templates.append(SRIGraphMatchingTemplate)
        self.templates.append(SRIVertexNominationTemplate)

        self.templates.append(BBNAudioClassificationTemplate)
        self.templates.append(SRICollaborativeFilteringTemplate)
        self.templates.append(DefaultTimeSeriesForcastingTemplate)
        self.templates.append(CMUClusteringTemplate)
        self.templates.append(MichiganVideoClassificationTemplate)

        self.templates.append(JHUVertexNominationTemplate)
        self.templates.append(JHUGraphMatchingTemplate)

        self.templates.append(ISIGraphNormClf)
        self.templates.append(ISI_GCN)

        # templates used for datasets with a large number of columns
        self.templates.append(Large_column_number_with_numerical_only_classification)
        self.templates.append(Large_column_number_with_numerical_only_regression)

        self.templates.append(ClassificationWithSelection)
        self.templates.append(RegressionWithSelection)

        # dsbox all in one templates
        # move dsboxClassificationTemplate to last execution because sometimes this template have bugs
        self.templates.append(dsboxClassificationTemplate)
        self.templates.append(dsboxRegressionTemplate)

        self._validate_templates(self.templates)

    def _load_single_inline_templates(self, template_name):
        if template_name in self.all_templates:
            self.templates.append(self.all_templates[template_name])
        else:
            raise KeyError("Template not found, name: {}".format(template_name))

    def _validate_templates(self, templates: typing.List[typing.Type[DSBoxTemplate]]) -> None:
        names: set = set()
        for template_class in templates:
            template = template_class()
            if template.template['name'] in names:
                raise ValueError(f'Multiple templates have the same name: {template.template["name"]}')
            names.add(template.template['name'])
            template.validate()



################################################################################################################


######################################            Templates            #########################################


################################################################################################################


################################################################################################################
##################################### General Classification Templates #########################################
################################################################################################################

################################################################################################
# A classification template encompassing several algorithms
################################################################################################
class DefaultClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() +
                     TemplateSteps.dsbox_feature_selector("classification",
                                                          first_input='data',
                                                          second_input='target') +
                     [
                         {
                             "name": "model_step",
                             "runtime": {
                                 "cross_validation": 5,
                                 "stratified": True
                             },
                             "primitives": [
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.random_forest.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'bootstrap': [True, False],
                                            'max_depth': [15, 30, None],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'n_estimators': [10, 50, 100],
                                         }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.extra_trees.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'bootstrap': [True, False],
                                            'max_depth': [15, 30, None],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'n_estimators': [10, 50, 100],
                                         }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.gradient_boosting.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'max_depth': [2, 3, 4, 5],
                                            'n_estimators': [50, 60, 80, 100],
                                            'learning_rate': [0.1, 0.2, 0.4, 0.5],
                                            'min_samples_split': [2, 3],
                                            'min_samples_leaf': [1, 2],
                                         }
                                 },
                             ],
                             "inputs": ["feature_selector_step", "target"]
                         }
                     ]
        }


class TestDefaultClassificationTemplate(DSBoxTemplate):
    # By Kyao
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "test_default_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() +
                     TemplateSteps.dsbox_feature_selector("classification",
                                                          first_input='data',
                                                          second_input='target') +
                     [
                         {
                             "name": "model_step",
                             "runtime": {
                                 "cross_validation": 5,
                                 "stratified": True
                             },
                             "primitives": [
                                 # {
                                 #     "primitive":
                                 #         "d3m.primitives.classification.random_forest.SKlearn",
                                 #     "hyperparameters":
                                 #         {
                                 #            'use_semantic_types': [True],
                                 #            'return_result': ['new'],
                                 #            'add_index_columns': [True],
                                 #            'bootstrap': [True, False],
                                 #            'max_depth': [15, 30, None],
                                 #            'min_samples_leaf': [1, 2, 4],
                                 #            'min_samples_split': [2, 5, 10],
                                 #            'max_features': ['auto', 'sqrt'],
                                 #            'n_estimators': [10, 50, 100],
                                 #         }
                                 # },
                                 # {
                                 #     "primitive":
                                 #         "d3m.primitives.classification.extra_trees.SKlearn",
                                 #     "hyperparameters":
                                 #         {
                                 #            'use_semantic_types': [True],
                                 #            'return_result': ['new'],
                                 #            'add_index_columns': [True],
                                 #            'bootstrap': [True, False],
                                 #            'max_depth': [15, 30, None],
                                 #            'min_samples_leaf': [1, 2, 4],
                                 #            'min_samples_split': [2, 5, 10],
                                 #            'max_features': ['auto', 'sqrt'],
                                 #            'n_estimators': [10, 50, 100],
                                 #         }
                                 # },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.gradient_boosting.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'max_depth': [2, 3, 4, 5],
                                            'n_estimators': [50, 60, 80, 100],
                                            'learning_rate': [0.1, 0.2, 0.4, 0.5],
                                            'min_samples_split': [2, 3],
                                            'min_samples_leaf': [1, 2],
                                         }
                                 },
                             ],
                             "inputs": ["feature_selector_step", "target"]
                         }
                     ]
        }


class NaiveBayesClassificationTemplate(DSBoxTemplate):
    '''A template encompassing several NB methods'''
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "naive_bayes_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
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


class RandomForestClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "random_forest_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
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
                                "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'return_result': ['new'],
                                    'add_index_columns': [True],
                                    # 'bootstrap': [True, False],
                                    # 'max_depth': [15, 30, None],
                                    # 'min_samples_leaf': [1, 2, 4],
                                    # 'min_samples_split': [2, 5, 10],
                                    # 'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100]
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class ExtraTreesClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "extra_trees_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
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
                                "d3m.primitives.classification.extra_trees.SKlearn",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'return_result': ['new'],
                                    'add_index_columns': [True],
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100]
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class GradientBoostingClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "gradient_boosting_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
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
                                "d3m.primitives.classification.gradient_boosting.SKlearn",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'return_result': ['new'],
                                    'add_index_columns': [True],
                                    'max_depth': [2, 3, 5],
                                    'n_estimators': [50, 75, 100],
                                    'learning_rate': [0.1, 0.3, 0.5],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class SVCClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "svc_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
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
                                "d3m.primitives.classification.svc.SKlearn",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'return_result': ['new'],
                                    'add_index_columns': [True],
                                    'C': [0.8, 1.0, 1.2],
                                    'kernel': ['rbf', 'poly'],
                                    'degree': [2, 3, 4],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class ClassificationWithSelection(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "classification_with_feature_selection",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.human_steps() + TemplateSteps.dsbox_feature_selector("classification") + [
                {
                    "name": "model_step",
                    "primitives": [
                        # {
                        #     "primitive": "d3m.primitives.classification.sgd.SKlearn",
                        #     "hyperparameters": {
                        #         'use_semantic_types': [True],
                        #         'return_result': ['new'],
                        #         'add_index_columns': [True],
                        #         "loss": ['log', 'hinge', 'squared_hinge', 'perceptron'],
                        #         "alpha": [float(x) for x in np.logspace(-6, -1.004, 7)],
                        #         "l1_ratio": [float(x) for x in np.logspace(-9, -0.004, 7)],
                        #         "penalty": ['elasticnet', 'l2']
                        #     }
                        # },
                        {
                            "primitive":
                            "d3m.primitives.classification.gradient_boosting.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                'max_depth': [2, 5],
                                'n_estimators': [50, 100],
                                'learning_rate': [0.1, 0.3],
                                'min_samples_split': [2, 3],
                                'min_samples_leaf': [1, 2],
                            }
                        },

                             ],
                    "inputs": ["feature_selector_step", "extract_target_step"]
                }
            ]
        }


class UMASSClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "UMASS_classification_template",
            "taskSubtype": {TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() +
                     TemplateSteps.dsbox_feature_selector("classification",
                                                          first_input='data',
                                                          second_input='target') +
                     [
                         {
                             "name": "model_step",
                             "primitives": "d3m.primitives.classification.multilabel_classifier.DSBOX",
                             "inputs": ["data, target"]
                         }
                     ]
        }


class dsboxClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "dsbox_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING',
            # 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING',
            # 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING',
            # 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                *TemplateSteps.dsbox_preprocessing(
                    clean_name="clean_step",
                    target_name="extract_target_step"
                ),
                *TemplateSteps.dsbox_encoding(clean_name="clean_step",
                                              encoded_name="encoder_step"),

                *TemplateSteps.dsbox_imputer(encoded_name="encoder_step",
                                             impute_name="impute_step"),

                # *dimensionality_reduction(feature_name="impute_step",
                #                           dim_reduce_name="dim_reduce_step"),

                *TemplateSteps.classifier_model(feature_name="impute_step",
                                                target_name='extract_target_step'),
            ]
        }


################################################################################################################
####################################   General Regression Templates    #########################################
################################################################################################################

################################################################################################
# A regression template encompassing several algorithms
################################################################################################
class DefaultRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name, "NONE"},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": False
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters":
                                {
                                    'max_depth': [2, 3, 4, 5],
                                    'n_estimators': [100, 130, 165, 200],
                                    'learning_rate': [0.1, 0.23, 0.34, 0.5],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.regression.extra_trees.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class SVRRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "svr_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name, "NONE"},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
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
                            "d3m.primitives.regression.svr.SKlearn",
                            "hyperparameters":
                            {
                                'C': [0.8, 1.0, 1.2],
                                'kernel': ['rbf', 'poly'],
                                'degree': [2, 3, 4, 5],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class GradientBoostingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "gradient_boosting_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name, "NONE"},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters":
                                {
                                    'max_depth': [2, 3, 5],
                                    'n_estimators': [100, 150, 200],
                                    'learning_rate': [0.1, 0.3, 0.5],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class ExtraTreesRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "extra_trees_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name, "NONE"},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.regression.extra_trees.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class RandomForestRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "random_forest_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name, "NONE"},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class RegressionWithSelection(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "regression_with_feature_selection",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.human_steps() + TemplateSteps.dsbox_feature_selector("regression") +
                     [
                         {
                             "name": "model_step",
                             "primitives": [
                                 {
                                     "primitive": "d3m.primitives.regression.sgd.SKlearn",
                                     "hyperparameters": {
                                         "loss": ['squared_loss', 'huber'],
                                         "alpha": [float(x) for x in np.logspace(-5, -1.004, 7)],  # cannot reach 0.1
                                         "l1_ratio": [0.01, 0.15, 0.3, 0.5, 0.6, 0.7, 0.9],  # cannot reach 1
                                         "learning_rate": ['optimal', 'invscaling'],
                                         'add_index_columns': [True],
                                         'use_semantic_types':[True],
                                     }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.regression.gradient_boosting.SKlearn",
                                     "hyperparameters":
                                         {
                                             'max_depth': [2, 3, 5],
                                             'n_estimators': [100, 150, 200],
                                             'learning_rate': [0.1, 0.3, 0.5],
                                             'min_samples_split': [2, 3],
                                             'min_samples_leaf': [1, 2],
                                             'add_index_columns': [True],
                                             'use_semantic_types':[True],
                                         }
                                 },
                             ],
                             "inputs": ["feature_selector_step", "extract_target_step"]
                         }
                     ]
        }


class DataAugmentRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)

        self.template = {
            "name": "data_augment_regression_template",
            "taskType": TaskType.REGRESSION.name,
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name, "NONE"},
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
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
                "name": "extract_attribute_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "extract_target_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "target_process_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["extract_target_step"]
            },
            {
                "name": "profile_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX"
                ],
                "inputs": ["profile_step"]
            },
            {
                "name": "encode_text_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.feature_construction.corex_text.DSBOX",
                        "hyperparameters":
                            {
                                'n_hidden': [(10)],
                                'threshold': [(500)],
                                'n_grams': [(1)],
                                'max_df': [(.9)],
                                'min_df': [(.02)],
                            }
                    },
                ],
                "inputs": ["clean_step"]
            },
            {
                "name": "encoder_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.label_encoder.DSBOX"
                ],
                "inputs": ["encode_text_step"]
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                "inputs": ["encode_text_step"]
            },
            {
                "name": "model_step",
                "primitives": [
                    {
                        "primitive" : "d3m.primitives.regression.gradient_boosting.SKlearn",
                        "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                            }
                    },
                ],
                "inputs": ["encode_text_step", "target_process_step"]
            },


            ]
        }


class dsboxRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "dsbox_regression_template",
            "taskType": TaskType.REGRESSION.name,
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name, "NONE"},
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                *TemplateSteps.dsbox_preprocessing(
                    clean_name="clean_step",
                    target_name="extract_target_step"
                ),
                *TemplateSteps.dsbox_encoding(
                    clean_name="clean_step",
                    encoded_name="encoder_step"
                ),
                *TemplateSteps.dsbox_imputer(
                    encoded_name="encoder_step",
                    impute_name="impute_step"
                ),
                *TemplateSteps.regression_model(
                    feature_name="impute_step",
                    target_name="extract_target_step"
                ),
            ]
        }



################################################################################################################
#####################################   Templates for Large Datasets  ##########################################
################################################################################################################


class Large_column_number_with_numerical_only_classification(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Large_column_number_with_numerical_only_classification",
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskSubtype.MULTICLASS.name, TaskSubtype.BINARY.name},
            "inputType": {"table", "large_column_number"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                # {
                #     "name": "encode1_step",
                #     # "primitives": ["d3m.primitives.data_preprocessing.unary_encoder.DSBOX","d3m.primitives.data_preprocessing.do_nothing.DSBOX"],
                #     "primitives": ["d3m.primitives.data_preprocessing.unary_encoder.DSBOX"],
                #     "inputs": ["extract_attribute_step"]
                # },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.data_cleaning.label_encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["encode2_step"],
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }],
                    "inputs": ["to_numeric_step", "extract_target_step"]
                },
            ]
        }


class Large_column_number_with_numerical_only_regression(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Large_column_number_with_numerical_only_regression",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            "inputType": {"table", "large_column_number"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
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

                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.data_cleaning.label_encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["encode2_step"],
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["to_numeric_step", "extract_target_step"]
                },
            ]
        }


class AlphaZeroEvalTemplate(DSBoxTemplate): # this is a template from succeed pipeline for uu2 dataset
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Alpha_Zero_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_prediction_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps":[
                {
                    "name": "denormalize_step", # step 0
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step", # step 1
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "column_parser_step", # step 2
                    "primitives": ["d3m.primitives.data_transformation.column_parser.DataFrameCommon"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_attribute_step", # step 3
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                            "hyperparameters": {
                                    'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                    'use_columns': (),
                                    'exclude_columns': ()
                                }
                        }
                    ],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "encoder_step", # step 4
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.one_hot_encoder.SKlearn",
                            "hyperparameters":{
                                "handle_unknown": ("ignore",)
                            }
                        }
                    ],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "cast_step", # step 5
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.cast_to_type.Common",
                            "hyperparameters":{
                                "type_to_case": ("float",)
                            }
                        }
                    ],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step", # step 6
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                            "hyperparameters": {
                                    'semantic_types': (
                                        'https://metadata.datadrivendiscovery.org/types/Target',
                                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'
                                    ),
                                    'use_columns': (),
                                    'exclude_columns': ()
                            }
                        }
                    ],
                    "inputs":["column_parser_step"]
                },
                {
                    "name": "cast_step_target", # step 7
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.cast_to_type.Common"
                        }
                    ],
                    "inputs": ["extract_target_step"]
                },
                {
                    "name": "model_step", # step 8
                    "primitives":[
                        {
                            "primitive": "d3m.primitives.regression.ridge.SKlearn"
                        }
                    ],
                    "inputs": ["cast_step", "cast_step_target"]
                },
                {
                    "name": "construct_prediction_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.DataFrameCommon",
                        }
                    ],
                    "inputs": ["model_step", "column_parser_step"]
                }
            ]
        }


class TESTINGTemplate(DSBoxTemplate): # this is a template from succeed pipeline for uu3 dataset
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Testing_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_prediction_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "update_semantic_step", # step 0
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.update_semantic_types.DatasetCommon",
                            "hyperparameters": {
                                "add_columns": [(1), (2), (3), (4), (5)],
                                "add_tpyes": ("https://metadata.datadrivendiscovery.org/types/CategoricalData",),
                                "resource_id": ("learningData", )
                            }
                        }
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "denormalize_step", # step 1
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.denormalize.Common"
                        }
                    ],
                    "inputs": ["update_semantic_step"]
                },
                {
                    "name": "to_dataframe_step", # step 2
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
                        }
                    ],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "column_parser_step", # step 3
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.column_parser.DataFrameCommon"
                        }
                    ],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_attribute_step", # step 4
                    "primitives":[
                        {
                            "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                            "hyperparameters": {
                                    'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                    'use_columns': (),
                                    'exclude_columns': ()
                                }
                        }
                    ],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "extract_target_step", # step 5
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                            "hyperparameters": {
                                    'semantic_types': (
                                        'https://metadata.datadrivendiscovery.org/types/Target',
                                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'
                                    ),
                                    'use_columns': (),
                                    'exclude_columns': ()
                                }
                        }
                    ],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "impute_step", # step 6
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_cleaning.imputer.SKlearn",
                        }
                    ],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "model_step", # step 7
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters": {
                                "return_result": ("replace", ),
                                "use_semanctic_types": [(True)],
                            }
                        }
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                },
                {
                    "name": "construct_predict_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.DataFrameCommon",
                        }
                    ],
                    "inputs": ["model_step", "to_dataframe_step"]
                }

            ]
        }
################################################################################################################
#####################################   TimeSeriesForcasting Templates  ########################################
################################################################################################################


class DefaultTimeSeriesForcastingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_Time_Series_Forcasting_Template",
            "taskType": TaskType.TIME_SERIES_FORECASTING.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": "NONE",
            "inputType": {"table", "timeseries"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
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
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profile_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": [
                        "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                        "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                    ],
                    "inputs": ["profile_step"]
                },
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.data_preprocessing.time_series_to_list.DSBOX"],
                    "inputs": ["clean_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX",
                            "hyperparameters":{
                                'generate_metadata':[True],
                            }
                        }
                    ],
                    "inputs": ["timeseries_to_list_step"]
                },

                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["extract_target_step"],
                },
                {
                    "name": "random_forest_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["random_projection_step", "to_numeric_step"]
                },
            ]
        }

class ARIMATemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "ARIMA_Template",
            "taskType": TaskType.TIME_SERIES_FORECASTING.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": "NONE",
            "inputType": {"table", "timeseries"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "ARIMA_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.normalization.denormalize.DSBOX"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                             'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                        "hyperparameters": {
                            "drop_non_numeric_columns": [(False)]
                        }
                    }],
                    "inputs": ["pre_extract_target_step"]
                },
                {
                    "name": "pre_extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs": ["pre_extract_attribute_step"]
                },
                {
                    "name": "ARIMA_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.time_series_forecasting.arima.DSBOX",
                            "hyperparameters": {
                                "take_log": [(False)],
                                "auto": [(True)]
                            }
                        }
                    ], # can add tuning parameters like: auto-fit, take_log, etc
                    "inputs": ["extract_attribute_step", "extract_target_step"]
                },
            ]
        }

class TimeSeriesForcastingTestingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TimeSeries_Forcasting_Testing_emplate",
            "taskType": TaskType.TIME_SERIES_FORECASTING.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": "NONE",
            "inputType": {"table", "timeseries"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "data_clean_step",
                    "primitives": ["d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["data_clean_step"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
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
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.extra_trees.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["encoder_step", "extract_target_step"]
                },
            ]
        }
'''
This template cannot run because of our templates' "input"/"output" schema
'''


class TimeSeriesForcastingTestingTemplate2(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TimeSeries_Forcasting_Testing_emplate",
            "taskType": TaskType.TIME_SERIES_FORECASTING.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": "NONE",
            "inputType": {"table", "timeseries"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': (3,)
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_time_series_file_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/FileName',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.data_preprocessing.time_series_to_list.DSBOX"],
                    "inputs": ["extract_time_series_file_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX",
                            "hyperparameters":{
                                'generate_metadata':[True],
                            }
                        }
                    ],
                    "inputs": ["timeseries_to_list_step"]
                },

                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "data_clean_step",
                    "primitives": ["d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["data_clean_step"]
                },
                {
                    "name": "concat_step",
                    "primitives": ["d3m.primitives.data.HorizontalConcatPrimitive"],
                    "inputs": ["encoder_step", "random_projection_step"]
                },
                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.extra_trees.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["concat_step", "extract_target_step"]
                },
            ]
        }

################################################################################################################
#####################################   ObjectDetectionTemplates   #############################################
################################################################################################################


class DefaultObjectDetectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DefaultObjectDetectionTemplate",
            "taskType": TaskType.OBJECT_DETECTION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": "NONE",
            "inputType": {"table", "image"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",#step 1
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                # read X value
                {
                    "name": "extract_file_step",#step 2
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
                                    'https://metadata.datadrivendiscovery.org/types/FileName',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",# step 3
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step", # step 4
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.yolo.DSBOX",
                            "hyperparameters": {
                            }
                        }
                    ],
                    "inputs": ["extract_file_step", "extract_target_step"]
                },
            ]
        }
################################################################################################################
#####################################   VideoClassificationTemplate   #############################################
################################################################################################################


class DefaultVideoClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DefaultVideoClassificationTemplate",
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": TaskSubtype.MULTICLASS.name,
            "inputType": "video",
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",#step 1
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                # read X value
                {
                    "name": "extract_file_step",#step 2
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/FileName',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",# step 3
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "video_reader",#step 4
                    "primitives": ["d3m.primitives.data_preprocessing.video_reader.DataFrameCommon"],
                    "inputs": ["extract_file_step"]
                },
                {
                    "name": "video_feature_extract",#step 5
                    "primitives": [
                            {
                                "primitive": "d3m.primitives.feature_extraction.inceptionV3_image_feature.DSBOX",
                                "hyperparameters": {
                                    "use_limitation":[True, False],
                                }
                            }

                        ],
                    "inputs": ["video_reader"]
                },
                {
                    "name": "model_step", # step 6
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.lstm.DSBOX",
                            "hyperparameters": {
                                "LSTM_units":[512,1024,2048],
                                "epochs":[10,100,1000],
                            }
                        }
                    ],
                    "inputs": ["video_feature_extract", "extract_target_step"]
                },
            ]
        }


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
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
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
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data_transformation.column_parser.DataFrameCommon"],
                #     "inputs": ["extract_target_step"]
                # },

                # read X value
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.data_preprocessing.time_series_to_list.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX",
                            "hyperparameters":{
                                'generate_metadata':[True],
                            }
                        }
                    ],
                    "inputs": ["timeseries_to_list_step"]
                },

                {
                    "name": "random_forest_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["random_projection_step", "extract_target_step"]
                },
            ]
        }


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
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
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
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.data_preprocessing.time_series_to_list.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX",
                            "hyperparameters":{
                                'generate_metadata':[True],
                            }
                        }
                    ],
                    "inputs": ["timeseries_to_list_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["extract_target_step"],
                },
                {
                    "name": "random_forest_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["random_projection_step", "to_numeric_step"]
                },
            ]
        }

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
                #     "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                #     "inputs": ["template_input"]
                # },
                # {
                #     "name": "to_dataframe_step",
                #     "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                #     "inputs": ["denormalize_step"]
                # },
                # # read Y value
                # {
                #     "name": "extract_target_step",
                #     "primitives": [{
                #         "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                #         "hyperparameters":
                #             {
                #                 'semantic_types': (
                #                 'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
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

################################################################################################################
#####################################   ImageProblemsTemplates   ###############################################
################################################################################################################


class TA1VggImageProcessingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1VggImageProcessingRegressionTemplate",
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "regressor_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
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
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.vgg16_image_feature.DSBOX",
                            "hyperparameters": {
                                'generate_metadata': [True]
                            }
                        }
                    ],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.pca.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types': [True]
                            }
                        }
                    ],
                    "inputs": ["feature_extraction"]
                },
                {
                    "name": "regressor_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                            }
                        }
                    ],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
            ]
        }


class DefaultImageProcessingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_image_processing_regression_template",
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "regressor_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
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
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data_transformation.column_parser.DataFrameCommon"],
                #     "inputs": ["to_dataframe_step"]
                # },
                # read X value
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.resnet50_image_feature.DSBOX",
                            "hyperparameters": {
                                'generate_metadata': [True]
                            }
                        }
                    ],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.pca.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types': [True]
                            }
                        }
                    ],
                    "inputs": ["feature_extraction"]
                },

                {
                    "name": "regressor_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
            ]
        }
################################################################################################################
#####################################   TextProblemsTemplates   ################################################
################################################################################################################


class DefaultTextClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_text_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": {"text", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_text_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                            "d3m.primitives.classification.gradient_boosting.SKlearn",
                            "hyperparameters":
                            {
                                'max_depth': [2, 5],
                                'n_estimators': [50, 100],
                                'learning_rate': [0.1, 0.3],
                                'min_samples_split': [2, 3],
                                'min_samples_leaf': [1, 2],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                        {
                            "primitive":
                            "d3m.primitives.classification.multinomial_naive_bayes.SKlearn",
                            "hyperparameters":
                            {
                                'alpha': [0, .5, 1],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                        {
                            "primitive":
                            "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class DefaultTextRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_text_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": {"text", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_text_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                    },
                    "primitives": [
                        {
                            "primitive":
                            "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters":
                            {
                                'max_depth': [2, 5],
                                'n_estimators': [100, 200],
                                'learning_rate': [0.1, 0.3],
                                'min_samples_split': [2, 3],
                                'min_samples_leaf': [1, 2],
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                        {
                            "primitive":
                            "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


################################################################################################################
#####################################   GraphProblemsTemplates   ###############################################
################################################################################################################


class DefaultLinkPredictionTemplate(DSBoxTemplate):
    '''
    Dummy implementation that does not look at the underlying graph at all.
    '''
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_LinkPrediction_Template",
            "taskType": {TaskType.LINK_PREDICTION.name, TaskType.GRAPH_MATCHING.name, TaskType.VERTEX_NOMINATION.name},
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["extract_attribute_step"],
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                        "hyperparameters": {
                            # 'bootstrap': [True, False],
                            'max_depth': [15, 30, None],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10],
                            'max_features': ['auto', 'sqrt'],
                            'n_estimators': [10, 50, 100],
                            'add_index_columns': [True],
                            'use_semantic_types':[True],
                        }
                    }
                    ],
                    "inputs": ["to_numeric_step", "extract_target_step"]
                }
            ]
        }

class SRIGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_GraphMatching_Template",
            "taskType": {TaskType.GRAPH_MATCHING.name, TaskType.LINK_PREDICTION.name},
            # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.link_prediction.graph_matching_link_prediction.GraphMatchingLinkPrediction",
                            "hyperparameters": {
                                "link_prediction_hyperparams": [(TemplateSteps.class_hyperparameter_generator(
                                    "d3m.primitives.link_prediction.graph_matching_link_prediction.GraphMatchingLinkPrediction", "link_prediction_hyperparams",
                                    {"truth_threshold": 0.0000001, "psl_options": "", "psl_temp_dir": "/tmp/psl/run",
                                     "postgres_db_name": "psl_d3m", "admm_iterations": 1000, "max_threads": 0,
                                     "jvm_memory": 0.75, "prediction_column": "match"}))]
                            }
                        }
                    ],
                    "inputs": ["template_input"]
                }
            ]
        }



class SRIVertexNominationTemplate(DSBoxTemplate):
    # not used for DS01876
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Vertex_Nomination_Template",
            "taskType": TaskType.VERTEX_NOMINATION.name,
            "taskSubtype": "NONE",
            "inputType": {"graph", "edgeList"},
            "output": "model_step",
            "steps": [
                {
                    "name": "parse_step",
                    "primitives": ["d3m.primitives.data_transformation.vertex_nomination_parser.VertexNominationParser"],
                    "inputs": ["template_input"]

                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.classification.vertex_nomination.VertexNomination"],
                    "inputs": ["parse_step"]

                }
            ]
        }


class SRICollaborativeFilteringTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Collaborative_Filtering_Template",
            "taskType": {TaskType.COLLABORATIVE_FILTERING.name},
            "taskSubtype": "NONE",
            "inputType": "table",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.link_prediction.collaborative_filtering_link_prediction.CollaborativeFilteringLinkPrediction"],
                    "inputs": ["template_input"]

                }
            ]
        }

class SRICommunityDetectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Community_Detection_Template",
            "taskType": {TaskType.COMMUNITY_DETECTION.name},
            "taskSubtype": {TaskSubtype.NONOVERLAPPING.name, TaskSubtype.OVERLAPPING.name},
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "parser_step",
                    "primitives": ["d3m.primitives.community_detection.community_detection_parser.CommunityDetectionParse"],
                    "inputs": ["template_input"]

                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.classification.community_detection.CommunityDetection"],
                    "inputs": ["parser_step"]
                }
            ]
        }

'''
JHU needs R supports and their primitives seem to be failing in pickling
'''


class JHUVertexNominationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Vertex_Nomination_Template",
            "taskType": TaskType.VERTEX_NOMINATION.name,
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "readgraph_step",
                    "primitives": [
                        "d3m.primitives.data_preprocessing.largest_connected_component.JHU"
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "embedding_step",
                    "primitives": [
                        "d3m.primitives.data_transformation.adjacency_spectral_embedding.JHU"
                    ],
                    "inputs": ["readgraph_step"]

                },
                {
                    "name": "model_step",
                    "primitives": [
                        "d3m.primitives.data_transformation.adjacency_spectral_embedding.JHU"
                    ],
                    "inputs": ["embedding_step"]
                }
            ]
        }

class JHUGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Graph_Matching_Template",
            "taskType": TaskType.GRAPH_MATCHING.name,
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.graph_matching.seeded_graph_matching.JHU"],
                    "inputs": ["template_input"]
                }
            ]
        }
################################################################################################################
#####################################   AudioClassificationTemplate   ##########################################
################################################################################################################


class BBNAudioClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "BBN_Audio_Classification_Template",
            "taskType": {TaskType.CLASSIFICATION.name},
            "taskSubtype": {TaskSubtype.MULTICLASS.name},
            "inputType": "audio",
            "output": "model_step",
            "steps": [
                # {
                #     "name": "denormalize_step",
                #     "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                #     "inputs": ["template_input"]
                # },
                # {
                #     "name": "to_dataframe_step",
                #     "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                #     "inputs": ["denormalize_step"]
                # },
                # {
                #     "name": "readtarget_step",
                #     "primitives": [{
                #         "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                #         "hyperparameters":
                #             {
                #                 'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                #                 'use_columns': (),
                #                 'exclude_columns': ()
                #             }
                #     }],
                #     "inputs": ["to_dataframe_step"]
                # },
                {
                    "name": "readtarget_step",
                    "primitives":["d3m.primitives.bbn.time_series.TargetsReader"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "readaudio_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.bbn.time_series.AudioReader",
                        "hyperparameters":
                            {
                                "read_as_mono": [(True)],
                                "resampling_rate": [(16000.0)],
                            }
                    }],
                    "inputs": ["template_input"]
                },
                {
                    "name": "channel_step",
                    "primitives": ["d3m.primitives.bbn.time_series.ChannelAverager"],
                    "inputs": ["readaudio_step"]
                },
                {
                    "name": "signaldither_step",
                    "primitives": [{"primitive": "d3m.primitives.bbn.time_series.SignalDither",
                                    "hyperparameters": {
                                        "level": [(0.0001)],
                                        "reseed": [(True)]
                                    }
                                    }],
                    "inputs": ["channel_step"]
                },
                {
                    "name": "signalframer_step",
                    "primitives": [{"primitive": "d3m.primitives.bbn.time_series.SignalFramer",
                                    "hyperparameters": {
                                        "flatten_output": [(False)],
                                        "frame_length_s": [(0.025)],
                                        "frame_shift_s": [(0.01)]
                                    }
                                    }],
                    "inputs": ["signaldither_step"]
                },
                {
                    "name": "MFCC_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.bbn.time_series.SignalMFCC",
                        "hyperparameters": {
                            "cep_lifter": [(22.0)],
                            "frame_mean_norm": [(False)],
                            "nfft": [(None)],
                            "num_ceps": [(20)],
                            "num_chans": [(20)],
                            "preemcoef": [(None)],
                            "use_power": [(False)]
                        }
                    }],
                    "inputs": ["signalframer_step"]
                },
                {
                    "name": "vectorextractor_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.bbn.time_series.IVectorExtractor",
                        "hyperparameters": {
                            "gmm_covariance_type": [("diag")],
                            "ivec_dim": [(100)],
                            "max_gmm_iter": [(20)],
                            "num_gauss": [(32)],
                            "num_ivec_iter": [(7)]
                        }
                    }],
                    "inputs": ["MFCC_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier",
                        "hyperparameters": {
                            "activation": [("relu")],
                            "add_index_columns": [(True)],
                            "alpha": [(0.0001)],
                            "beta_1": [(0.9)],
                            "beta_2": [(0.999)],
                            "early_stopping": [(True)],
                            "epsilon": [(1e-8)],
                            "exclude_columns": [([])],
                            # "hidden_layer_sizes":[([30,30])],
                            "learning_rate": [("constant")],
                            "learning_rate_init": [(0.01)],
                            "max_iter": [(200)],
                            "return_result": [("replace")],
                            "shuffle": [(True)],
                            "solver": [("adam")],
                            "tol": [(0.0001)],
                            "use_columns": [([])],
                            "use_semantic_types": [(False)],
                            "warm_start": [(False)]
                        }
                    }],
                    "inputs": ["vectorextractor_step", "readtarget_step"]

                }
            ]
        }

################################################################################################################
#####################################   SRIMeanbaselineTemplate   ##############################################
################################################################################################################


class SRIMeanBaselineTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Mean_Baseline_Template",
            "taskType": "NONE",
            "taskSubtype": "NONE",
            "inputType": "NONE",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.classification.gaussian_classification.MeanBaseline"],
                    "inputs": ["template_input"]

                }
            ]
        }

################################################################################################################
#####################################   ClusteringTemplate   ###################################################
################################################################################################################


class CMUClusteringTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.need_add_reference = True
        self.template = {
            "name": "CMU_Clustering_Template",
            "taskType": TaskType.CLUSTERING.name,
            "taskSubtype": "NONE",
            "inputType": "table",
            "output": "output_step",
            "steps": [
                {
                    "name": "to_dataframe_step", # step 0
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "column_parser_step",# step 1
                    "primitives": ["d3m.primitives.data_transformation.column_parser.DataFrameCommon"],
                    "inputs":["to_dataframe_step"],
                },

                {
                    "name": "extract_attribute_step", # step 2
                    "primitives": ["d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon"],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "model_step", # step 3
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.cover_tree.Fastlvm",
                            "hyperparameters": {
                                "k": [(1), (4), (6), (8), (10), (12)]
                            }
                        }
                    ],
                    "inputs": ["data_clean_step"]
                },
                {
                    "name": "output_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.DataFrameCommon",
                        }
                    ],
                    "inputs": ["model_step", "column_parser_step"]
                }
            ]
        }


################################################################################################################
#####################################   VideoClassificationTemplate   ##########################################
################################################################################################################


class MichiganVideoClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Michigan_Video_Classification_Template",
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": TaskSubtype.MULTICLASS.name,
            "inputType": "video",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "construct_prediction_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "read_video_step",
                    "primitives": ["d3m.primitives.data_preprocessing.video_reader.DataFrameCommon"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "featurize_step",
                    "primitives": ["d3m.primitives.feature_extraction.i3d.Umich"],
                    "inputs": ["read_video_step"]

                },
                {
                    "name": "convert_step",
                    "primitives": ["d3m.primitives.data_transformation.ndarray_to_dataframe.Common"],
                    "inputs": ["featurize_step"]

                },
                {
                    "name": "model_step",
                    # "primitives": ["d3m.primitives.classifier.RandomForest"],
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }],
                    "inputs": ["convert_step", "extract_target_step"]
                },
                {
                    "name": "construct_prediction_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.construct_predictions.DataFrameCommon",
                        }
                    ],
                    "inputs": ["model_step", "to_dataframe_step"]
                }
            ]
        }


################################################################################################################
#####################################   TA1Template   ##########################################################
################################################################################################################

class TA1ClassificationTemplate1(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1_classification_template_1",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
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
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute'),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
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


class TA1Classification_2(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1Classification_2",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "text",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                *TemplateSteps.default_dataparser(attribute_name="extract_attribute_step",
                                                  target_name="extract_target_step"),
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.data_cleaning.label_encoder.DSBOX"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "nothing_step",
                    "primitives": ["d3m.primitives.data_preprocessing.do_nothing.DSBOX"],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives": ["d3m.primitives.normalization.iqr_scaler.DSBOX"],
                    "inputs": ["nothing_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["scaler_step", "extract_target_step"]
                }
            ]
        }


class TA1Classification_3(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1Classification_3",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
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
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profile_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": [
                        "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                        "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                    ],
                    "inputs": ["profile_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "label_step",
                    "primitives": ["d3m.primitives.data_cleaning.label_encoder.DSBOX"],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
                    "inputs": ["label_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }],
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": False
                    },
                    "inputs": ["corex_step", "extract_target_step"]
                }
            ]
        }

    # @override

class MuxinTA1ClassificationTemplate1(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MuxinTA1ClassificationTemplate1",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
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
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "encode1_step",
                    "primitives": ["d3m.primitives.data_preprocessing.unary_encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["encode1_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
                    "inputs": ["encode2_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["corex_step"],
                },
                {
                    "name": "impute_step",
                    # "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "primitives": ["d3m.primitives.data_preprocessing.greedy_imputation.DSBOX"],
                    # "primitives": ["d3m.primitives.data_preprocessing.iterative_regression_imputation.DSBOX"],

                    "inputs": ["to_numeric_step", "extract_target_step" ]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 2,
                        # "cross_validation":1,
                        "stratified": True
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                                {
                                    'max_depth': [(2), (4)],  # (10), #
                                    'n_estimators': [(10), (30)],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override

class MuxinTA1ClassificationTemplate2(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MuxinTA1ClassificationTemplate2",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
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
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
                    "inputs": ["encode_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["corex_step"],
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.greedy_imputation.DSBOX"],
                    "inputs": ["to_numeric_step", "extract_target_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": False
                    },
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }


class MuxinTA1ClassificationTemplate3(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MuxinTA1ClassificationTemplate3",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
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
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "no_op_step",
                    "primitives": ["d3m.primitives.data_preprocessing.do_nothing.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["no_op_step"]
                },
                {
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
                    "inputs": ["encode_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["corex_step"],
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.iterative_regression_imputation.DSBOX"],
                    "inputs": ["to_numeric_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "runtime": {
                        "cross_validation": 5,
                        "stratified": False
                    },
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

'''
# unfinished!
class DefaultImageClassificationWithCNNTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_Image_classification_with_CNN_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": {"table", "image"},
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
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                                    'https://metadata.datadrivendiscovery.org/types/FileName',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "encode1_step",
                    "primitives": ["d3m.primitives.data_preprocessing.unary_encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["encode1_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["encode2_step"],
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "inputs": ["to_numeric_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                        "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                    }
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }
'''

class UU3TestTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "UU3_Test_Template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            "inputType": "table",
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "multi_table_processing_step",
                    "primitives": ["d3m.primitives.feature_extraction.multitable_featurization.DSBOX"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["multi_table_processing_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["multi_table_processing_step"]
                },
                {
                    "name": "encode1_step",
                    "primitives": ["d3m.primitives.data_preprocessing.unary_encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["encode1_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["encode2_step"],
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "inputs": ["to_numeric_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                        "hyperparameters":
                            {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                    }
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }



################################################################################################################
#####################################   HorizontalTemplate   ###################################################
################################################################################################################


class HorizontalTemplate(DSBoxTemplate): #This template only generate processed features
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Horizontal_Template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name, TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": {TaskType.CLASSIFICATION.name, TaskType.REGRESSION.name},
            "inputType": "table",
            "output": "scaler_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                                    ),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": [
                        "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                    ],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_construction.corex_text.DSBOX",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": [
                        "d3m.primitives.data_preprocessing.encoder.DSBOX",
                    ],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.normalization.iqr_scaler.DSBOX",
                            "hyperparameters": {}
                        },
                    ],
                    "inputs": ["impute_step"]
                },
                # {
                #     "name": "extract_target_step",
                #     "primitives": [{
                #         "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                #         "hyperparameters":
                #             {
                #                 'semantic_types': (
                #                     #'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                #                     'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                #                 'use_columns': (),
                #                 'exclude_columns': ()
                #             }
                #     }],
                #     "inputs": ["to_dataframe_step"]
                # },
            ]
        }

class ISIGraphNormClf(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "ISI_graph_norm_clf",
            "taskType": {TaskType.COLLABORATIVE_FILTERING.name, TaskType.VERTEX_NOMINATION.name, TaskType.COMMUNITY_DETECTION.name, TaskType.LINK_PREDICTION.name},
            "taskSubtype": {"NONE", TaskSubtype.NONOVERLAPPING.name, TaskSubtype.OVERLAPPING.name},
            #"taskSubtype": "NONE",
            #"inputType": "table",
            "inputType": {"graph","table"},
            "output": "model_step",
            "steps": [
                {
                    "name": "readgraph_step",
                    "primitives": [
                        "d3m.primitives.data_transformation.normalize_graphs.Common"
                        #"d3m.primitives.data_preprocessing.largest_connected_component.JHU"
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_graph_tables",#_learning",
                    "primitives": ["d3m.primitives.data_transformation.graph_to_edge_list.DSBOX"],
                    "inputs": ["readgraph_step"]
                },
                {
                    "name": "to_learning_dataframe",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                              {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',
                                                   'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'),
                                                   #'https://metadata.datadrivendiscovery.org/types/PrimaryKey'),
                                'use_columns': (),  #'d3mIndex'),
                                'exclude_columns': ()  #[[1]]
                            }
                    }],
                    "inputs": ["to_learning_dataframe"]  #_learning"]
                    },
                    {
                    "name": "embedding_step",
                    "primitives": [
                        "d3m.primitives.feature_construction.graph_transformer.SDNE"
                    ],
                    "hyperparameters": {
                        "return_list": [False]
                    },
                    "inputs": ["extract_graph_tables"]#["to_dataframe_nodes", "to_dataframe_edges"] #["readgraph_step"]
                },
                {
                     "name": "model_step",
                     "primitives": [{
                     "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                     "hyperparameters": {
                         # 'bootstrap': [True, False],
                         'max_depth': [15, 30, None],
                         'min_samples_leaf': [1, 2, 4],
                         'min_samples_split': [2, 5, 10],
                         'max_features': ['auto', 'sqrt'],
                         'n_estimators': [10, 50, 100],
                         'add_index_columns': [True],
                         'use_semantic_types':[False],
                         'error_on_no_input':[True],
                         #'exclude_input_columns': [[1]]
                         #'exclude_output_columns': ['nodeID']
                         }
                     }
                    ],
                #{
                #    "name": "model_step",
                #    "primitives": [

                #
                #        #"d3m.primitives.classification.gaussian_classification.JHU"
                #    ],
                #        "d3m.primitives.classification.gaussian_classification.JHU"
                #    ],
                    "inputs": ["embedding_step", "extract_target_step"]
                }
            ]
        }



class ISI_GCN(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
	    "name": "ISI_gcn",
            "taskType": {TaskType.COLLABORATIVE_FILTERING.name, TaskType.VERTEX_NOMINATION.name, TaskType.COMMUNITY_DETECTION.name, TaskType.LINK_PREDICTION.name},
            "taskSubtype": {"NONE", TaskSubtype.NONOVERLAPPING.name, TaskSubtype.OVERLAPPING.name},
            #"taskSubtype": "NONE",
            #"inputType": "table",
            "inputType": {"graph", "table"},
	        "output": "model_step",
            "steps": [
                {
		    "name": "readgraph_step",
                    "primitives": [
                        "d3m.primitives.data_transformation.normalize_graphs.Common"
                        #"d3m.primitives.data_preprocessing.largest_connected_component.JHU"
                    ],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_graph_tables",#_learning",
                    "primitives": ["d3m.primitives.data_transformation.graph_to_edge_list.DSBOX"],
                    "inputs": ["readgraph_step"]
                },
              {
                    "name": "extract_graph_tables",#_learning",
                    "primitives": ["d3m.primitives.data_transformation.graph_to_edge_list.DSBOX"],
                    "inputs": ["readgraph_step"]
                },
                {
                    "name": "to_learning_dataframe",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_learning_dataframe"]#_learning"]
                },
                {
                    #"name": "model_step", #
                    "name": "embedding_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_construction.graph_transformer.GCN"
                        }
                    ],
                    "inputs": ["extract_graph_tables", "extract_target_step"]
                },
                {
                     "name": "model_step",
                     "primitives": [{
                         "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                         "hyperparameters": {
                             # 'bootstrap': [True, False],
                             'max_depth': [15, 30, None],
                             'min_samples_leaf': [1, 2, 4],
                             'min_samples_split': [2, 5, 10],
                             'max_features': ['auto', 'sqrt'],
                             'n_estimators': [10, 50, 100],
                             'add_index_columns': [True],
                             'use_semantic_types':[False],
                             'error_on_no_input':[True]
                         }
                     }
                    ],
                # #{
                # #    "name": "model_step",
                # #    "primitives": [

                # #
                # #        #"d3m.primitives.classification.gaussian_classification.JHU"
                # #    ],
                # #        "d3m.primitives.classification.gaussian_classification.JHU"
                # #    ],
                    "inputs": ["embedding_step", "extract_target_step"]
                }
            ]
        }

class LupiPriviledgedInformationClassification(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "LupiPrivilegedInfoCls",
            "taskType": {TaskType.CLASSIFICATION.name},
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "inputType": {"table"},
            "specializedProblem": {SpecializedProblem.PRIVILEGED_INFORMATION},
            "output": "prediction_step",
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
                    "name": "model_step",
                    "primitives": [{
                        "primitive":  "d3m.primitives.classification.lupi_svm.LupiSvmClassifier",
                        "hyperparameters": {
                            "C": [1],
                            "C_gridsearch": [(-4.0, 26.0, 0.3)],
                            "add_index_columns": [False],
                            "class_weight": ['balanced'],
                            "coef0": [0],
                            "degree": [3],
                            "gamma": ["auto"],
                            "gamma_gridsearch": [(-4.0, 26.0, 0.3)],
                            "kernel": ["rbf"],
                            "max_iter": [-1],
                            "n_jobs": [4],
                            "probability": [False],
                            "return_result": ["new"],
                            "shrinking": [True],
                            "tol": [0.001],
                            "use_semantic_types": [False],
                        }
                    },
                    ],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "prediction_step",
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.DataFrameCommon"],
                    "inputs": ["model_step", "to_dataframe_step"]
                }

            ]
        }
