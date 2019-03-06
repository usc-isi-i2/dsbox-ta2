import logging

_logger = logging.getLogger(__name__)


class TemplateSteps:

    '''
    Some steps and parameters that are used for creating templates
    Returns a list of dicts with the most common steps
    '''
    @staticmethod
    def dsbox_generic_steps(data: str = "data", target: str = "target"):
        '''
        dsbox generic step for classification and regression, directly lead to model step
        '''
        return [
            {
                "name": "sampling_step",
                "primitives": ["d3m.primitives.data_preprocessing.DoNothingForDataset.DSBOX"],
                "inputs": ["template_input"]
            },

            {
                "name": "denormalize_step",
                "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                "inputs": ["sampling_step"]
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
                                ),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.Profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.CleaningFeaturizer.DSBOX",
                    # "d3m.primitives.data_preprocessing.DoNothing.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encode_step",
                "primitives": ["d3m.primitives.data_preprocessing.Encoder.DSBOX"],
                "inputs": ["clean_step"]
            },
            {
                "name": "corex_step",
                "primitives": ["d3m.primitives.feature_construction.corex_text.CorexText"],
                "inputs": ["encode_step"]
            },
            {
                "name": "to_numeric_step",
                "primitives": ["d3m.primitives.data_transformation.ToNumeric.DSBOX"],
                "inputs":["corex_step"],
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.MeanImputation.DSBOX"],
                "inputs": ["to_numeric_step"]
            },
            {
                "name": "scaler_step",
                "primitives": [
                    # {
                    #     "primitive": "d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn",
                    #     "hyperparameters":
                    #     {
                    #         'use_semantic_types':[True],
                    #         'return_result':['new'],
                    #         'add_index_columns':[True],
                    #     }
                    # },
                    {
                        "primitive": "d3m.primitives.normalization.IQRScaler.DSBOX",
                        "hyperparameters": {}
                    },
                    "d3m.primitives.data_preprocessing.DoNothing.DSBOX",
                ],
                "inputs": ["impute_step"]
            },
            {
                "name": data,
                "primitives": [
                    # 19 Feb 2019: Stop using PCA until issue is resolved
                    # https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues/154
                    # {
                    #     "primitive": "d3m.primitives.data_transformation.pca.SKlearn",
                    #     "hyperparameters":
                    #     {
                    #         'use_semantic_types': [True],
                    #         'add_index_columns': [True],
                    #         'return_result': ['new'],
                    #         'n_components': [10, 15, 25]
                    #     }
                    # },
                    "d3m.primitives.data_preprocessing.DoNothing.DSBOX",
                ],
                "inputs": ["scaler_step"]
            },
            {
                "name": "pre_"+target,
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
                "name": target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.ToNumeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["pre_"+target]
            },
        ]

    # Returns a list of dicts with the most common steps
    @staticmethod
    def dsbox_generic_text_steps(data: str = "data", target: str = "target"):
        '''
        dsbox generic step for classification and regression, specify on text data processing, directly lead to model step
        '''
        return [
            {
                "name": "sampling_step",
                "primitives": ["d3m.primitives.data_preprocessing.DoNothingForDataset.DSBOX"],
                "inputs": ["template_input"]
            },
            {
                "name": "denormalize_step",
                "primitives": [
                    "d3m.primitives.data_transformation.denormalize.Common"
                ],
                "inputs": ["sampling_step"]
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
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.Profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.CleaningFeaturizer.DSBOX",
                    "d3m.primitives.data_preprocessing.DoNothing.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encoder_step",
                "primitives": [
                    "d3m.primitives.data_preprocessing.Encoder.DSBOX",
                    "d3m.primitives.data_preprocessing.DoNothing.DSBOX"
                ],
                "inputs": ["clean_step"]
            },
            {
                "name": "corex_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.feature_construction.corex_text.CorexText",
                        "hyperparameters":
                            {
                                'n_hidden': [5, 10],
                                'threshold': [0, 500],
                                'n_grams': [1, 3],
                            }
                    },
                ],
                "inputs": ["encoder_step"]
            },

            {
                "name": "to_numeric_step",
                "primitives": ["d3m.primitives.data_transformation.ToNumeric.DSBOX"],
                "inputs":["corex_step"],
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.MeanImputation.DSBOX"],
                "inputs": ["to_numeric_step"]
            },
            {
                "name": data,
                "primitives": [
                    # Feb 17, 2019: Do not use until issue is resolved
                    # https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues/153
                    # {
                    #     "primitive": "d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn",
                    #     "hyperparameters": {
                    #         'add_index_columns': [True],
                    #         'use_semantic_types': [True],
                    #     }
                    # },
                    {
                        "primitive": "d3m.primitives.normalization.IQRScaler.DSBOX",
                        "hyperparameters": {}
                    },
                    "d3m.primitives.data_preprocessing.DoNothing.DSBOX",
                ],
                "inputs": ["impute_step"]
            },
            {
                "name": "pre_"+target,
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
                "name": target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.ToNumeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["pre_"+target]
            },
        ]

    @staticmethod
    def human_steps():
        '''
        simulated solution steps for classification and regression
        '''
        return [
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
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.Profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.CleaningFeaturizer.DSBOX",
                    "d3m.primitives.data_preprocessing.DoNothing.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encoder_step",
                "primitives": [
                    "d3m.primitives.data_preprocessing.Encoder.DSBOX",
                    "d3m.primitives.data_cleaning.Labeler.DSBOX"
                ],
                "inputs": ["clean_step"]
            },
            {
                "name": "to_numeric_step",
                "primitives": ["d3m.primitives.data_transformation.ToNumeric.DSBOX"],
                "inputs":["encoder_step"],
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.MeanImputation.DSBOX"],
                "inputs": ["to_numeric_step"]
            },
            {
                "name": "pre_extract_target_step",
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
                "name": "extract_target_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.ToNumeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["pre_extract_target_step"]
            },
        ]

    @staticmethod
    def dsbox_feature_selector(ptype, first_input='impute_step', second_input='extract_target_step'):
        import numpy as np
        '''
        dsbox feature selection steps for classification and regression, lead to feature selector steps
        '''
        if ptype == "regression":
            return [
                {
                    "name": "feature_selector_step",
                    "primitives": [
                        {
                            # 1 March 2019: select_fwe disappeared from sklearn wrap
                            # "primitive": "d3m.primitives.feature_selection.select_fwe.SKlearn",
                            # "hyperparameters": {
                            #     'use_semantic_types': [True],
                            #     'add_index_columns': [True],
                            #     'return_result': ['new'],
                            #     'add_index_columns': [True],
                            #     "alpha": [float(x) for x in np.logspace(-4, -1, 6)]
                            # }
                        },
                        {
                            "primitive": "d3m.primitives.feature_selection.generic_univariate_select.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                "score_func": ["f_regression"],
                                "mode": ["percentile"],
                                "param": [5, 7, 10, 15, 30, 50, 75],
                            }
                        },
                        "d3m.primitives.data_preprocessing.DoNothing.DSBOX"
                    ],
                    "inputs":[first_input, second_input]
                },
            ]
        else:
            return [
                {
                    "name": "feature_selector_step",
                    "primitives": [
                        # 1 March 2019: select_fwe disappeared from sklearn wrap
                        # {
                        #     "primitive": "d3m.primitives.feature_selection.select_fwe.SKlearn",
                        #     "hyperparameters": {
                        #         'use_semantic_types': [True],
                        #         'return_result': ['new'],
                        #         'add_index_columns': [True],
                        #         "alpha": [float(x) for x in np.logspace(-4, -1, 6)]
                        #     }
                        # },

                        {
                            "primitive": "d3m.primitives.feature_selection.generic_univariate_select.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                "mode": ["percentile"],
                                "param": [5, 7, 10, 15, 30, 50, 75],
                            }
                        },
                        "d3m.primitives.data_preprocessing.DoNothing.DSBOX"

                    ],
                    "inputs":[first_input, second_input]
                },
            ]

    @staticmethod
    def dimensionality_reduction(feature_name: str = "impute_step",
                                 dim_reduce_name: str = "dim_reduce_step"):
        '''
        dsbox dimension reduction steps for classification and regression
        '''
        return \
            [
                {
                    "name": dim_reduce_name,
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_preprocessing.DoNothing.DSBOX",
                        },
                        # 19 Feb 2019: Stop using PCA until issue is resolved
                        # https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues/154
                        # {
                        #     "primitive": "d3m.primitives.data_transformation.pca.SKlearn",
                        #     "hyperparameters":
                        #         {
                        #             'add_index_columns': [True],
                        #             'use_semantic_types':[True],
                        #             'n_components': [(2), (4), (8), (16), (32), (64), (128)], }
                        # },
                        {
                            "primitive": "d3m.primitives.data_transformation.kernel_pca.SKlearn",
                            "hyperparameters":
                                {
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                    'n_components': [(2), (4), (8), (16), (32), (64), (128)],
                                    'kernel': [('rbf'), ('sigmoid'), ('cosine')]
                                }
                        },
                        {
                            "primitive": "d3m.primitives.data_transformation.kernel_pca.SKlearn",
                            "hyperparameters":
                                {
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                    'n_components': [(2), (4), (8), (16), (32), (64), (128)],
                                    'kernel': [('poly')],
                                    'degree': [(2), (3), (4)],
                                }
                        },
                        {
                            "primitive": "d3m.primitives.feature_selection.variance_threshold.SKlearn",
                            "hyperparameters":
                                {
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                    'threshold': [(0.01), (0.01), (0.05), (0.1)],
                                }
                        },
                    ],
                    "inputs": [feature_name]
                }
            ]
    '''
    Other steps models
    '''

    @staticmethod
    def default_dataparser(attribute_name: str = "extract_attribute_step",
                           target_name: str = "extract_target_step"):
        return \
            [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ['template_input']
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": attribute_name,
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
                    "name": "pre_"+target_name,
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
                    "name": target_name,
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.ToNumeric.DSBOX",
                        "hyperparameters": {
                            "drop_non_numeric_columns": [False]
                        }
                    }],
                    "inputs": ["pre_"+target_name]
                }
            ]

    @staticmethod
    def d3m_preprocessing(attribute_name: str = "cast_1_step",
                          target_name: str = "extract_target_step"):
        return \
            [
                *TemplateSteps.default_dataparser(target_name=target_name),
                {
                    "name": "column_parser_step",
                    "primitives": ["d3m.primitives.data_transformation.column_parser.DataFrameCommon"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": attribute_name,
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.ToNumeric.DSBOX",
                            "hyperparameters": {}
                        },
                        {
                            "primitive": "d3m.primitives.data_preprocessing.DoNothing.DSBOX",
                            "hyperparameters": {}
                        }

                    ],
                    "inputs": ["column_parser_step"]
                },
            ]

    @staticmethod
    def dsbox_preprocessing(clean_name: str = "clean_step",
                            target_name: str = "extract_target_step"):
        return \
            [
                *TemplateSteps.default_dataparser(target_name=target_name),
                {
                    "name": "profile_step",
                    "primitives": ["d3m.primitives.schema_discovery.Profiler.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": clean_name,
                    "primitives": [
                        "d3m.primitives.data_cleaning.CleaningFeaturizer.DSBOX",
                        "d3m.primitives.data_preprocessing.DoNothing.DSBOX"
                    ],
                    "inputs": ["profile_step"]
                },
            ]

    @staticmethod
    def dsbox_encoding(clean_name: str = "clean_step",
                       encoded_name: str = "cast_1_step"):
        return \
            [
                # TODO the ColumnParser primitive is buggy as it generates arbitrary nan values
                # {
                #     "name": "encode_strings_step",
                #     "primitives": ["d3m.primitives.data_transformation.column_parser.DataFrameCommon"],
                #     "inputs": [clean_name]
                # },
                {
                    "name": "encode_text_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_construction.corex_text.CorexText",
                            "hyperparameters":
                                {
                                    'n_hidden': [(10)],
                                    'threshold': [(0), (500)],
                                    'n_grams': [(1), (5)],
                                    'max_df': [(.9)],
                                    'min_df': [(.02)],
                                }
                        },
                        {"primitive": "d3m.primitives.data_preprocessing.DoNothing.DSBOX", },
                    ],
                    "inputs": [clean_name]
                },
                # {
                #     "name": "encode_unary_step",
                #     "primitives": [
                #         {"primitive": "d3m.primitives.data_preprocessing.UnaryEncoder.DSBOX", },
                #         {"primitive": "d3m.primitives.data_preprocessing.DoNothing.DSBOX", },
                #     ],
                #     "inputs": ["encode_text_step"]
                # },
                {
                    # "name": 'encode_string_step',
                    "name": encoded_name,
                    "primitives": [
                        {"primitive": "d3m.primitives.data_preprocessing.Encoder.DSBOX", },
                        {"primitive": "d3m.primitives.data_cleaning.Labeler.DSBOX", },
                        # {"primitive": "d3m.primitives.data_preprocessing.DoNothing.DSBOX", },
                    ],
                    "inputs": ["encode_text_step"]
                },
            ]

    @staticmethod
    def dsbox_imputer(encoded_name: str = "cast_1_step",
                      impute_name: str = "impute_step"):
        return \
            [
                {
                    "name": "base_impute_step",
                    "primitives": [
                        {"primitive": "d3m.primitives.data_preprocessing.MeanImputation.DSBOX", },
                        # {"primitive": "d3m.primitives.data_preprocessing.GreedyImputation.DSBOX", },
                        {"primitive": "d3m.primitives.data_preprocessing.MeanImputation.DSBOX", },
                        {"primitive": "d3m.primitives.data_preprocessing.IterativeRegressionImputation.DSBOX", },
                        # {"primitive": "d3m.primitives.data_preprocessing.DoNothing.DSBOX", },
                    ],
                    "inputs": [encoded_name]
                },
                {
                    "name": impute_name,
                    "primitives": [
                        # Feb 17, 2019: Do not use until issue is resolved
                        # https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues/153
                        # {
                        #     "primitive": "d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn",
                        #     "hyperparameters": {
                        #         'use_semantic_types': [True],
                        #         'add_index_columns': [True],
                        #     }},
                        {
                            "primitive": "d3m.primitives.normalization.IQRScaler.DSBOX",
                            "hyperparameters": {
                                "use_semantic_types": [True],
                                "add_index_columns": [True],
                            }
                        },

                        # !!!! KYAO
                        # {"primitive": "d3m.primitives.data_preprocessing.DoNothing.DSBOX", },
                    ],
                    "inputs": ["base_impute_step"]
                },
            ]

    @staticmethod
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
                            "d3m.primitives.classification.random_forest.SKlearn",
                        "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                                'max_depth': [(2), (4), (8)],  # (10), #
                                'n_estimators': [(2), (5), (10), (20), (30), (40)]
                            }
                    },
                        {
                            "primitive":
                                "d3m.primitives.classification.svc.SKlearn",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'add_index_columns': [True],
                                    'C': [(1), (10), (100)],  # (10), #
                                }
                    }, {
                            "primitive":
                                "d3m.primitives.classification.multinomial_naive_bayes.SKlearn",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'add_index_columns': [True],
                                    'alpha': [(1)],
                                }
                    },
                    ],
                    "inputs": [feature_name, target_name]
                }
            ]

    @staticmethod
    def regression_model(feature_name: str = "impute_step",
                         target_name: str = "extract_target_step"):
        return \
            [
                {
                    "name": "model_step",
                    "primitives": [
                        # linear ridge : simplistic linear regression with L2 regularization
                        {
                            "primitive": "d3m.primitives.regression.ridge.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                                'alpha': [(1)],
                            }},
                        # Least-angle regression: (1 / (2 * n_samples)) * |y - Xw|^2_2 + alpha * |w|_1
                        {
                            "primitive": "d3m.primitives.regression.lars.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                            }},
                        # Nearest Neighbour
                        {
                            "primitive": "d3m.primitives.regression.k_neighbors.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                            }},
                        # Support Vector Regression Method
                        {
                            "primitive": "d3m.primitives.regression.svr.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                            }},

                        {
                            "primitive": "d3m.primitives.regression.sgd.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                                'alpha': [(1)],
                            }},
                        {
                            "primitive": "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters":
                            {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                            }},
                    ],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": [feature_name, target_name]
                }
            ]

    @staticmethod
    def class_hyperparameter_generator(primitive_name, parameter_name, definition):
        from d3m import index
        g = None
        try:
            g = index.get_primitive(primitive_name).metadata.query()["primitive_code"]["hyperparams"][parameter_name]['structural_type'](definition)
        except Exception:
            _logger.error(f"Hyperparameter not valid for {primitive_name}!")
            pass
        return g
