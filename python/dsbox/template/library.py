import json
import glob
import typing

from d3m import index
from d3m.container.dataset import SEMANTIC_TYPES
from d3m.metadata.problem import TaskType, TaskSubtype

from dsbox.template.template import TemplatePipeline, DSBoxTemplate


class TemplateDescription:
    """
    Description of templates in the template library.

    Attributes
    ----------
    task : TaskType
        The type task the template handles
    template: TemplatePipeline
        The actual template
    target_step: int
        The step of the template that extract the ground truth target from
        the dataset
    predicted_target_step: int
        The step of the template generates the predictions
    """

    def __init__(self, task: TaskType, template: TemplatePipeline,
                 target_step: int, predicted_target_step: int) -> None:
        self.task = task
        self.template = template

        # Instead of having these attributes here, probably should attach
        # attributes to the template steps
        self.target_step = target_step
        self.predicted_target_step = predicted_target_step


class TemplateLibrary:
    """
    Library of template pipelines
    """

    def __init__(self, library_dir: str = None, run_single_template: str = "") -> None:
        self.templates: typing.List[typing.Type[DSBoxTemplate]] = []
        self.primitive: typing.Dict = index.search()

        self.library_dir = library_dir
        if self.library_dir is None:
            self._load_library()

        self.all_templates = {
            'random_forest_template': RandomForestTemplate, 
            "Default_classification_template": DefaultClassificationTemplate,
            "dsbox_classification_template": dsboxClassificationTemplate,
            "Default_regression_template": DefaultRegressionTemplate,
            "Default_timeseries_collection_template": DefaultTimeseriesCollectionTemplate,
            "Default_image_processing_regression_template":
                DefaultImageProcessingRegressionTemplate,
            "TA1DefaultImageProcessingRegressionTemplate":
                TA1DefaultImageProcessingRegressionTemplate,
            "TA1_classification_template_1": TA1ClassificationTemplate1,
            "MuxinTA1ClassificationTemplate1": MuxinTA1ClassificationTemplate1,
            "MuxinTA1ClassificationTemplate2": MuxinTA1ClassificationTemplate2,
            "MuxinTA1ClassificationTemplate3": MuxinTA1ClassificationTemplate3,
            "MuxinTA1ClassificationTemplate4": MuxinTA1ClassificationTemplate4,
            "UU3_Test_Template": UU3TestTemplate,
            "TA1Classification_2": TA1Classification_2,
            "TA1Classification_3": TA1Classification_3,
            "TA1VggImageProcessingRegressionTemplate": TA1VggImageProcessingRegressionTemplate,
            "Default_text_classification_template": DefaultTextClassificationTemplate, 
            "BBN_audio_classification_template": BBNAudioClassificationTemplate, 
            "SRI_GraphMatching_Template":SRIGraphMatchingTemplate,
            "SRI_Vertex_Nomination_Template":SRIVertexNominationTemplate, 
            "SRI_Collaborative_Filtering_Template":SRICollaborativeFilteringTemplate, 
            "SRI_Community_Detection_Template":SRICommunityDetectionTemplate, 
            "UCHI_Time_Series_Classification_Template":UCHITimeSeriesClassificationTemplate
        }

        if run_single_template:
            self._load_single_inline_templates(run_single_template)
        else:
            self._load_inline_templates()

    def get_templates(self, task: TaskType, subtype: TaskSubtype, taskSourceType: SEMANTIC_TYPES)\
            -> \
    typing.List[DSBoxTemplate]:
        results = []
        for template_class in self.templates:
            template = template_class()
            # sourceType refer to d3m/container/dataset.py ("SEMANTIC_TYPES" as line 40-70)
            # taskType and taskSubtype refer to d3m/
            if task.name in template.template['taskType'] and subtype.name in template.template[
                'taskSubtype']:
                # if there is only one task source type which is table, we don't need to check
                # other things
                if {"table"} == taskSourceType and template.template['inputType'] == "table":
                    results.append(template)
                else:
                    # otherwise, we need to process in another way because "table" source type
                    # exist nearly in every dataset
                    if "table" in taskSourceType:
                        taskSourceType.remove("table")

                    for each_source_type in taskSourceType:
                        if each_source_type in {template.template['inputType']}:
                            results.append(template)
        # if we finally did not find a proper template to use
        if results == []:
            print("Error: Can't find a suitable template type to fit the problem.")
        else:
            print("[INFO] Template choices:")
            # otherwise print the template list we added
            for each_template in results:
                print("Template '", each_template.template["name"],
                      "' has been added to template base.")
        return results

    def _load_library(self):
        # TODO
        # os.path.join(library_dir, 'template_library.yaml')
        pass

    def _load_inline_templates(self):

        # self.templates.append(RandomForestTemplate)
        # Tabular Classification
        self.templates.append(DefaultRegressionTemplate)
        self.templates.append(dsboxRegressionTemplate)

        # Tabular Regression
        self.templates.append(DefaultClassificationTemplate)
        self.templates.append(dsboxClassificationTemplate)
        self.templates.append(TA1Classification_3)
        self.templates.append(MuxinTA1ClassificationTemplate1)
        self.templates.append(TA1ClassificationTemplate1)

        # Image Regression
        self.templates.append(DefaultImageProcessingRegressionTemplate)
        self.templates.append(TA1DefaultImageProcessingRegressionTemplate)

        # Others
        self.templates.append(DefaultTimeseriesCollectionTemplate)
        self.templates.append(DefaultTextClassificationTemplate)
        self.templates.append(SRIGraphMatchingTemplate)
        self.templates.append(SRIVertexNominationTemplate)
        self.templates.append(SRICommunityDetectionTemplate)
        self.templates.append(JHUVertexNominationTemplate)
        self.templates.append(BBNAudioClassificationTemplate)
        self.templates.append(SRICollaborativeFilteringTemplate)
        self.templates.append(UCHITimeSeriesClassificationTemplate)

    def _load_single_inline_templates(self, template_name):
        if template_name in self.all_templates:
            self.templates.append(self.all_templates[template_name])
        else:
            raise KeyError("Template not found, name: {}".format(template_name))


class SemanticTypeDict(object):
    def __init__(self, libdir):
        self.pos = libdir
        self.mapper = {}

    def read_primitives(self) -> None:

        # jsonPath = os.path.join(libdir, filename)
        # print(self.pos)
        user_Defined_Confs = glob.glob(
            self.pos + "/*_template_semantic_mapping.json")
        # print(user_Defined_Confs)
        for u in user_Defined_Confs:
            with open(u, "r") as cf:
                print("opened", u)
                for v in json.load(cf).items():
                    self.mapper[v[0]] = v[1]

    def create_configuration_space(self, template: TemplatePipeline):
        definition = {}
        # for t in TemplatePipeline:
        #     if isinstance(t, list):
        steps = template.template_nodes.keys()
        for s in steps:
            if template.template_nodes[s].semantic_type in self.mapper.keys():
                definition[s] = self.mapper[
                    template.template_nodes[s].semantic_type]

        # return SimpleConfigurationSpace(definition)
        return definition


def regression_model(feature_name: str = "impute_step",
                     target_name: str = "extract_target_step"):
    return \
        [
            {
                "name": "model_step",
                "primitives": [
                    # Bayesian ARD regression: based on Serban's work it's the most stable model
                    {"primitive": "d3m.primitives.sklearn_wrap.SKARDRegression", },
                    # linear ridge : simplistic linear regression with L2 regularization
                    {"primitive": "d3m.primitives.sklearn_wrap.SKRidge", },
                    # Least-angle regression: (1 / (2 * n_samples)) * |y - Xw|^2_2 + alpha * |w|_1
                    {"primitive": "d3m.primitives.sklearn_wrap.SKLars", },
                    # Nearest Neighbour
                    {"primitive": "d3m.primitives.sklearn_wrap.SKKNeighborsRegressor", },
                    # Support Vector Regression Method
                    {"primitive": "d3m.primitives.sklearn_wrap.SKLinearSVR", },

                    {"primitive": "d3m.primitives.sklearn_wrap.SKSGDRegressor", },
                    {"primitive": "d3m.primitives.sklearn_wrap.SKGradientBoostingRegressor", },
                ],
                "runtime": {
                    "cross_validation": 10,
                    "stratified": False
                },
                "inputs": [feature_name, target_name]
            }
        ]


def dimensionality_reduction(feature_name: str = "impute_step",
                            dim_reduce_name: str = "dim_reduce_step"):
    return \
        [
            {
                "name": dim_reduce_name,
                "primitives": [
                    {
                        "primitive": "d3m.primitives.sklearn_wrap.SKPCA",
                        "hyperparameters":
                            {'n_components': [(8), (16), (32), (64), (128)], }
                    },
                    {
                        "primitive": "d3m.primitives.sklearn_wrap.SKKernelPCA",
                        "hyperparameters":
                            {
                                'n_components': [(8), (16), (32), (64), (128)],
                                'kernel': [('rbf'), ('sigmoid'), ('cosine')]
                            }
                    },
                    {
                        "primitive": "d3m.primitives.sklearn_wrap.SKKernelPCA",
                        "hyperparameters":
                            {
                                'n_components': [(8), (16), (32), (64), (128)],
                                'kernel': [('poly')],
                                'degree': [(2), (3), (4)],
                            }
                    },
                    {
                        "primitive": "d3m.primitives.sklearn_wrap.SKVarianceThreshold",
                        "hyperparameters":
                            {'threshold': [(0), (0.01), (0.1), (0.15)], }
                    },
                ],
                "inputs": [feature_name]
            }
        ]


def classifier_model(feature_name: str = "impute_step",
                     target_name: str = "extract_target_step"):
    return \
        [
            {
                "name": "model_step",
                "runtime": {
                    "cross_validation": 10,
                    "stratified": True
                },
                "primitives": [{
                    "primitive":
                        "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                    "hyperparameters":
                        {
                            'max_depth': [(2), (4), (8)],  # (10), #
                            'n_estimators': [(10), (20), (30)]
                        }
                },
                    {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKLinearSVC",
                        "hyperparameters":
                            {
                                'C': [(1), (10), (100)],  # (10), #
                            }
                    }, {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKMultinomialNB",
                        "hyperparameters":
                            {
                                'alpha': [(1)],
                            }
                    },
                ],
                "inputs": [feature_name, target_name]
            }
        ]


def default_dataparser(attribute_name: str = "extract_attribute_step",
                       target_name: str = "extract_target_step"):
    return \
        [
            {
                "name": "denormalize_step",
                "primitives": ["d3m.primitives.dsbox.Denormalize"],
                "inputs": ['template_input']
            },
            {
                "name": "to_dataframe_step",
                "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                "inputs": ["denormalize_step"]
            },
            {
                "name": attribute_name,
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
                "name": target_name,
                "primitives": [{
                    "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["to_dataframe_step"]
            }
        ]


def d3m_preprocessing(attribute_name: str = "cast_1_step",
                      target_name: str = "extract_target_step"):
    return \
        [
            *default_dataparser(target_name=target_name),
            {
                "name": "column_parser_step",
                "primitives": ["d3m.primitives.data.ColumnParser"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": attribute_name,
                "primitives": ["d3m.primitives.data.CastToType"],
                "inputs": ["column_parser_step"]
            },
        ]


def dsbox_preprocessing(clean_name: str = "clean_step",
                        target_name: str = "extract_target_step"):
    return \
        [
            *default_dataparser(target_name=target_name),
            {
                "name": "profile_step",
                "primitives": ["d3m.primitives.dsbox.Profiler"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": clean_name,
                "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                "inputs": ["profile_step"]
            },
        ]


def dsbox_encoding(clean_name: str = "clean_step",
                   encoded_name: str = "cast_1_step"):
    return \
        [
            # TODO the ColumnParser primitive is buggy as it generates arbitrary nan values
            # {
            #     "name": "encode_strings_step",
            #     "primitives": ["d3m.primitives.data.ColumnParser"],
            #     "inputs": [clean_name]
            # },
            {
                "name": "encode_text_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.dsbox.CorexText",
                        "hyperparameters":
                            {
                                'n_hidden': [(10)],
                                'threshold': [(0)],
                                'threshold': [(0), (500)],
                                'n_grams': [(1), (5)],
                                'max_df': [(.9)],
                                'min_df': [(.02)],
                            }
                    },
                    {"primitive": "d3m.primitives.dsbox.DoNothing", },
                ],
                "inputs": [clean_name]
            },
            # {
            #     "name": "encode_unary_step",
            #     "primitives": [
            #         {"primitive": "d3m.primitives.dsbox.UnaryEncoder", },
            #         {"primitive": "d3m.primitives.dsbox.DoNothing", },
            #     ],
            #     "inputs": ["encode_text_step"]
            # },
            {
                # "name": 'encode_string_step',
                "name": encoded_name,
                "primitives": [
                    {"primitive": "d3m.primitives.dsbox.Encoder", },
                    {"primitive": "d3m.primitives.dsbox.Labler", },
                    #{"primitive": "d3m.primitives.dsbox.DoNothing", },
                ],
                "inputs": ["encode_text_step"]
            },

            # {
            #     "name": encoded_name,
            #     "primitives": [{
            #         "primitive": "d3m.primitives.data.CastToType",
            #         "hyperparameters":
            #             {
            #                 'type_to_cast': ['float','str'],
            #             }
            #     }],
            #     "inputs": ["encode_string_step"]
            # },
        ]


def dsbox_imputer(encoded_name: str = "cast_1_step",
                  impute_name: str = "impute_step"):
    return \
        [
            {
                "name": "base_impute_step",
                "primitives": [
                    {"primitive": "d3m.primitives.sklearn_wrap.SKImputer", },
                    {"primitive": "d3m.primitives.dsbox.GreedyImputation", },
                    {"primitive": "d3m.primitives.dsbox.MeanImputation", },
                    {"primitive": "d3m.primitives.dsbox.IterativeRegressionImputation", },
                    #{"primitive": "d3m.primitives.dsbox.DoNothing", },
                ],
                "inputs": [encoded_name]
            },
            {
                "name": impute_name,
                "primitives": [
                    {"primitive": "d3m.primitives.dsbox.IQRScaler", },
                    {"primitive": "d3m.primitives.dsbox.DoNothing", },
                ],
                "inputs": ["base_impute_step"]
            },
        ]


class RandomForestTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "random_forest_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
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
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                    # 'n_hidden':[(10)],
                                    # 'threshold':[(0)],
                                    # # 'threshold':[(0), (500)],
                                    # 'n_grams':[(1), (5)],
                                    # 'max_df':[(.9)],
                                    # 'min_df':[(.02)],
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        "hyperparameters":
                            {
                            'max_depth': [(2),(4),(8)],
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class DefaultClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
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
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                    # 'n_hidden':[(10)],
                                    # 'threshold':[(0)],
                                    # # 'threshold':[(0), (500)],
                                    # 'n_grams':[(1), (5)],
                                    # 'max_df':[(.9)],
                                    # 'min_df':[(.02)],
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        "hyperparameters":
                            {
                            # 'max_depth': [(2),(4),(8)], #(10), #
                            # 'n_estimators':[(10),(20)]
                            }
                        },
                        # {
                        # "primitive":
                        #     "d3m.primitives.sklearn_wrap.SKLinearSVC",
                        # "hyperparameters":
                        #     {
                        #     'C': [(1), (10), (100)],  # (10), #
                        #     }
                        # },
                        # {
                        #     "primitive":
                        #         "d3m.primitives.sklearn_wrap.SKMultinomialNB",
                        #     "hyperparameters":
                        #         {
                        #             'alpha': [(1)],
                        #         }
                        # },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class dsboxClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DSBox_classification_template",
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
                *dsbox_preprocessing(
                    clean_name="clean_step",
                    target_name="extract_target_step"
                ),
                *dsbox_encoding(clean_name="clean_step",
                                encoded_name="encoder_step"),

                # {
                #     "name": "columns_parser_step",
                #     "primitives": ["d3m.primitives.data.ColumnParser"],
                #     "inputs": ["encoder_step"]
                # },

                *dsbox_imputer(encoded_name="encoder_step",
                               impute_name="impute_step"),

                *classifier_model(feature_name="impute_step",
                                  target_name='extract_target_step'),
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class DefaultRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_regression_template",
            "taskType": TaskType.REGRESSION.name,
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
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
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                    # 'n_hidden':[(10)],
                                    # 'threshold':[(0)],
                                    # # 'threshold':[(0), (500)],
                                    # 'n_grams':[(1), (5)],
                                    # 'max_df':[(.9)],
                                    # 'min_df':[(.02)],
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
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
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKARDRegression",
                        "hyperparameters":
                            {
                            }
                    }, 
                    # {
                    #     "primitive":
                    #         "d3m.primitives.sklearn_wrap.SKSGDRegressor",
                    #     "hyperparameters":
                    #         {
                    #         }
                    # }, 
                    # {
                    #     "primitive":
                    #         "d3m.primitives.sklearn_wrap.SKGradientBoostingRegressor",
                    #     "hyperparameters":
                    #         {
                    #         }
                    # }
                    ],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["impute_step", "extract_target_step"]
                },
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class dsboxRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DSBox_regression_template",
            "taskType": TaskType.REGRESSION.name,
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                *dsbox_preprocessing(
                    clean_name="clean_step",
                    target_name="extract_target_step"
                ),
                *dsbox_encoding(
                    clean_name="clean_step",
                    encoded_name="encoder_step"
                ),
                *dsbox_imputer(
                    encoded_name="encoder_step",
                    impute_name="impute_step"
                ),
                *regression_model(
                    feature_name="impute_step",
                    target_name="extract_target_step"
                ),
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


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
            "inputType": "timeseries",  # See SEMANTIC_TYPES.keys() for range of values
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
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
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


class TA1VggImageProcessingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1VggImageProcessingRegressionTemplate",
            "taskType": TaskType.REGRESSION.name, # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype" : {TaskSubtype.UNIVARIATE.name,TaskSubtype.MULTIVARIATE.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "regressor_step",  # Name of the final step generating the prediction
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
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data.ColumnParser"],
                #     "inputs": ["to_dataframe_step"]
                # },
                # read X value
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.dsbox.DataFrameToTensor"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    "primitives": ["d3m.primitives.dsbox.ResNet50ImageFeature"],
                    # or "primitives": ["d3m.primitives.dsbox.Vgg16ImageFeature"],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKPCA"],
                    "inputs": ["feature_extraction"]
                },

                {
                    "name": "regressor_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestRegressor"],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
            ]
        }

    def importance(datset, problem_description):
        return 7


class TA1DefaultImageProcessingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_image_processing_regression_template",
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "regressor_step",  # Name of the final step generating the prediction
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
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/Target',
                                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data.ColumnParser"],
                #     "inputs": ["to_dataframe_step"]
                # },
                # read X value
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.dsbox.DataFrameToTensor"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    # "primitives": ["d3m.primitives.dsbox.ResNet50ImageFeature"],
                    "primitives": ["d3m.primitives.dsbox.Vgg16ImageFeature"],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKPCA"],
                    "inputs": ["feature_extraction"]
                },

                {
                    "name": "regressor_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestRegressor"],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
            ]
        }

    def importance(datset, problem_description):
        return 7


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
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
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
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encode_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [{
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        "hyperparameters":
                            {
                                'max_depth': [(2), (4), (8)],  # (10), #
                                'n_estimators': [(10), (20), (30)]
                            }
                    },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class DefaultTextClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_text_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "text",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
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
                    "name": "column_parser_step",
                    "primitives": ["d3m.primitives.data.ColumnParser"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "cast_1_step",
                    "primitives": ["d3m.primitives.data.CastToType"],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.dsbox.CorexText"],
                    "inputs": ["cast_1_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        # {
                        # "primitive":
                        #     "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        # "hyperparameters":
                        #     {
                        #     'max_depth': [(2),(4),(8)], #(10), #
                        #     'n_estimators':[(10),(20),(30)]
                        #     }
                        # },
                        # {
                        # "primitive":
                        #     "d3m.primitives.sklearn_wrap.SKLinearSVC",
                        # "hyperparameters":
                        #     {
                        #     'C': [(1), (10), (100)],  # (10), #
                        #     }
                        # },
                        {
                            "primitive":
                                "d3m.primitives.sklearn_wrap.SKMultinomialNB",
                            "hyperparameters":
                                {
                                    'alpha': [(1)],
                                }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7




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
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target',
                                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "encode1_step",
                    #"primitives": ["d3m.primitives.dsbox.UnaryEncoder","d3m.primitives.dsbox.DoNothing"],
                    "primitives": ["d3m.primitives.dsbox.UnaryEncoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder","d3m.primitives.dsbox.DoNothing"],
                    #"primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs":["encode1_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encode2_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 2,
                        #"test_validation":1,
                        "stratified": True
                    },
                    "primitives": [
                    {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        "hyperparameters":
                            {
                                'max_depth': [(2), (4)],  # (10), #
                                'n_estimators':[(10), (30)]
                            }
                    },
                    # {
                    #     "primitive":
                    #         "d3m.primitives.sklearn_wrap.SKLinearSVC",
                    #     "hyperparameters":
                    #         {
                    #             'C': [(1), (10)],  # (10), #
                    #         }
                    # }, 
                    # {
                    #     "primitive":
                    #         "d3m.primitives.sklearn_wrap.SKMultinomialNB",
                    #     "hyperparameters":
                    #         {
                    #             'alpha': [(1)],
                    #         }
                    # },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }
    # @override

    def importance(datset, problem_description):
        return 7


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
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "no_op_step",
                    "primitives": ["d3m.primitives.dsbox.DoNothing"],
                    "inputs":["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target',
                                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["no_op_step"]
                },
                {
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.GreedyImputation"],
                    "inputs": ["encode_step", "extract_target_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "cast_1_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        }
                    ],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["cast_1_step", "extract_target_step"]
                }
            ]
        }
    # @override

    def importance(datset, problem_description):
        return 7


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
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "no_op_step",
                    "primitives": ["d3m.primitives.dsbox.DoNothing"],
                    "inputs":["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target',
                                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["no_op_step"]
                },
                {
                    "name": "encode_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.IterativeRegressionImputation"],
                    "inputs": ["encode_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "cast_1_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data.CastToType",
                            "hyperparameters": {"type_to_cast": ["float"]}
                        }
                    ],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["cast_1_step", "extract_target_step"]
                }
            ]
        }
    # @override

    def importance(datset, problem_description):
        return 7


class MuxinTA1ClassificationTemplate4(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "MuxinTA1ClassificationTemplate4",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
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
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target',
                                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "encode1_step",
                    "primitives": ["d3m.primitives.dsbox.UnaryEncoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs":["encode1_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encode2_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "model_step",
                    "primitives": [
                        "d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }
    # @override

    def importance(datset, problem_description):
        return 7


class UU3TestTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "UU3_Test_Template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name,TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "multi_table_processing_step",
                    "primitives": ["d3m.primitives.dsbox.MultiTableFeaturization"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["multi_table_processing_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target',
                                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["multi_table_processing_step"]
                },
                {
                    "name": "encode1_step",
                    "primitives": ["d3m.primitives.dsbox.UnaryEncoder"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs":["encode1_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["encode2_step"]
                },
                {
                    "name": "cast_step",
                    "primitives": [
                    {"primitive":"d3m.primitives.data.CastToType",
                    "hyperparameters":{"type_to_cast":["float"]}
                    }
                    ],
                    "inputs": ["impute_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestRegressor",
                        "hyperparameters":
                            {
                            }
                    }
                    ],
                    "inputs": ["cast_step", "extract_target_step"]
                }
            ]
        }
    # @override
    def importance(datset, problem_description):
        return 7


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
                *default_dataparser(attribute_name="extract_attribute_step",
                                    target_name="extract_target_step"),
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.dsbox.CorexText"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Labler"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "nothing_step",
                    "primitives": ["d3m.primitives.dsbox.DoNothing"],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "scaler_step",
                    "primitives": ["d3m.primitives.dsbox.IQRScaler"],
                    "inputs": ["nothing_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "inputs": ["scaler_step", "extract_target_step"]
                }
            ]
        }
        # import pprint
        # pprint.pprint(self.template)
        # exit(1)

    # @override
    def importance(datset, problem_description):
        return 7


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
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target',
                                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profile_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs":["profile_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "label_step",
                    "primitives": ["d3m.primitives.dsbox.Labler"],
                    "inputs": ["impute_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": ["d3m.primitives.dsbox.CorexText"],
                    "inputs": ["label_step"]
                },
                # {
                #     "name": "corex_step",
                #     "primitives": ["d3m.primitives.dsbox.CorexText"],
                #     "inputs": ["cast_1_step"]
                # },
                {
                    "name": "model_step",
                    "primitives": [
                        "d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["corex_step", "extract_target_step"]
                }
            ]
        }
    # @override

    def importance(datset, problem_description):
        return 7

class DefaultImageProcessingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_image_processing_regression_template",
            "taskType": TaskType.REGRESSION.name, # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype" : {TaskSubtype.UNIVARIATE.name,TaskSubtype.MULTIVARIATE.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output" : "regressor_step",  # Name of the final step generating the prediction
            "target" : "extract_target_step",  # Name of the step generating the ground truth
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
                        "primitive":"d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target','https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                            }
                        }],
                    "inputs": ["to_dataframe_step"]
                },
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data.ColumnParser"],
                #     "inputs": ["to_dataframe_step"]
                # },
                # read X value
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.dsbox.DataFrameToTensor"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    "primitives": ["d3m.primitives.dsbox.Vgg16ImageFeature"],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKPCA"],
                    "inputs": ["feature_extraction"]
                },

                {
                    "name": "regressor_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestRegressor"],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
            ]
        }
    def importance(datset, problem_description):
        return 7



class SRILinkPredictionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_LinkPrediction_Template",
            "taskType": TaskType.LINK_PREDICTION.name,
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "parser_step",
                    "primitives": ["d3m.primitives.sri.graph.GraphMatchingParser"],
                    "inputs":["template_input"]
                },
                {
                    "name": "transform_step",
                    "primitives": ["d3m.primitives.sri.graph.GraphTransformer"],
                    "inputs":["parser_step"]
                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sri.psl.LinkPrediction"],
                    "inputs":["transform_step"]
                }]
        }

    def importance(datset, problem_description):
        return 7


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
                    # "primitives": [{
                    # "primitive":"d3m.primitives.sri.psl.GraphMatchingLinkPrediction", 
                    # "hyperparameters":{
                    # "truth_threshold":[(0.1),(0.5), (0.9)]
                    # }
                    # }],
                    "primitives":["d3m.primitives.sri.psl.GraphMatchingLinkPrediction"],
                    "inputs":["template_input"]
                }
            ]
        }

    def importance(dataset, problem_description):
        return 7


class SRIVertexNominationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "SRI_Vertex_Nomination_Template",
            "taskType": TaskType.VERTEX_NOMINATION.name,
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "parse_step",
                    "primitives": ["d3m.primitives.sri.graph.VertexNominationParser"],
                    "inputs":["template_input"]

                },
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sri.psl.VertexNomination"],
                    "inputs": ["parse_step"]

                }
            ]
        }

    def importance(datset, problem_description):
        return 7

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
                    "primitives": ["d3m.primitives.sri.psl.CollaborativeFilteringLinkPrediction"],
                    "inputs": ["template_input"]

                }
            ]
        }

    def importance(dataset, problem_description):
        return 7

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
                    "primitives": ["d3m.primitives.sri.graph.CommunityDetectionParser"],
                    "inputs": ["template_input"]

                },
                {
                    "name":"model_step",
                    "primitives":["d3m.primitives.sri.psl.CommunityDetection"], 
                    "inputs":["parser_step"]
                }
            ]
        }

    def importance(dataset, problem_description):
        return 7    



class JHUVertexNominationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Vertex_Nomination_Template",
            "taskType": {TaskType.VERTEX_NOMINATION.name},
            "taskSubtype": "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.jhu_primitives.SpectralGraphClustering"],
                    "inputs": ["template_input"]

                }
            ]
        }

    def importance(dataset, problem_description):
        return 7


class JHUGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "JHU_Graph_Matching_Template",
            "taskType": {TaskType.GRAPH_MATCHING.name},
            "taskSubtype": "None",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.jhu_primitives.SeededGraphMatching"],
                    "inputs":["template_input"]
                }
            ]
        }

    def importance(datset, problem_description):
        return 7



class BBNAudioClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
        "name" :"BBN_Audio_Classification_Template",
        "taskType":{TaskType.CLASSIFICATION.name},
        "taskSubtype":{TaskSubtype.MULTICLASS.name},
        "inputType":"audio",
        "output":"model_step",
        "steps":[
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
        {
            "name":"readtarget_step", 
            "primitives": [{
                "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                "hyperparameters":
                {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
            "inputs":["to_dataframe_step"]
        },

        {
            "name": "readaudio_step", 
            "primitives":[{
            "primitive": "d3m.primitives.bbn.time_series.AudioReader", 
            "hyperparameters":
            {
                "read_as_mono": [(True)],
                "resampling_rate":[(16000.0)], 
            }
            }],
            "inputs":["template_input"]        
        }, 
        {
            "name":"channel_step", 
            "primitives":["d3m.primitives.bbn.time_series.ChannelAverager"], 
            "inputs":["readaudio_step"]
        }, 
        {
            "name": "signaldither_step", 
            "primitives":[{"primitive":"d3m.primitives.bbn.time_series.SignalDither", 
            "hyperparameters":{
                "level":[(0.0001)], 
                "reseed":[(True)]
            }
            }], 
            "inputs":["channel_step"]
        }, 
        {
            "name":"signalframer_step", 
            "primitives":[{"primitive":"d3m.primitives.bbn.time_series.SignalFramer", 
            "hyperparameters":{
                "flatten_output":[(False)], 
                "frame_length_s":[(0.025)], 
                "frame_shift_s":[(0.01)]
            }
            }],
            "inputs":["signaldither_step"]
        }, 
        {
            "name":"MFCC_step", 
            "primitives":[{
            "primitive":"d3m.primitives.bbn.time_series.SignalMFCC", 
            "hyperparameters":{
                "cep_lifter": [(22.0)], 
                "frame_mean_norm":[(False)], 
                "nfft": [(None)], 
                "num_ceps":[(20)],
                "num_chans":[(20)], 
                "preemcoef":[(None)], 
                "use_power":[(False)]
            }
            }],
            "inputs":["signalframer_step"]
        }, 
        {
            "name":"vectorextractor_step", 
            "primitives":[{
            "primitive":"d3m.primitives.bbn.time_series.IVectorExtractor", 
            "hyperparameters":{
                "gmm_covariance_type":[("diag")], 
                "ivec_dim":[(100)], 
                "max_gmm_iter":[(20)], 
                "num_gauss":[(32)], 
                "num_ivec_iter":[(7)]
            }
            }],
            "inputs":["MFCC_step"]
        }, 
        {
            "name":"model_step", 
            "primitives":[{
            "primitive":"d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier", 
            "hyperparameters":{
                "activation":[("relu")], 
                "add_index_columns":[(True)], 
                "alpha":[(0.0001)], 
                "beta_1":[(0.9)], 
                "beta_2":[(0.999)], 
                "early_stopping":[(True)], 
                "epsilon":[(1e-8)], 
                "exclude_columns":[([])], 
                # "hidden_layer_sizes":[([30,30])], 
                "learning_rate":[("constant")], 
                "learning_rate_init":[(0.01)], 
                "max_iter":[(200)], 
                "return_result":[("replace")], 
                "shuffle":[(True)],
                "solver":[("adam")], 
                "tol":[(0.0001)], 
                "use_columns":[([])], 
                "use_semantic_types":[(False)],
                "warm_start":[(False)]
            }
            }],
            "inputs":["vectorextractor_step", "readtarget_step"]

        }
        ]
        }
        
    def importance(datset, problem_description):
        return 7



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
                "name":"model_step", 
                "primitives":["d3m.primitives.datasmash.d3m_XG2"], 
                "inputs":["template_input", "template_input"]
            }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7
