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
                "primitives": ["d3m.primitives.dsbox.Profiler"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    # "d3m.primitives.dsbox.CleaningFeaturizer",
                    "d3m.primitives.dsbox.DoNothing",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "corex_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.dsbox.CorexText",
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
                    "d3m.primitives.dsbox.Encoder",
                    "d3m.primitives.dsbox.DoNothing"
                ],
                "inputs": ["corex_step"]
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                "inputs": ["encoder_step"]
            },
            {
                "name": "scaler_step",
                "primitives": [
                    # {
                    #     "primitive": "d3m.primitives.sklearn_wrap.SKMaxAbsScaler",
                    #     "hyperparameters": 
                    #     {
                    #         'use_semantic_types':[True],
                    #         'return_result':['new'],
                    #         'add_index_columns':[True],
                    #     }
                    # },
                    {
                        "primitive": "d3m.primitives.dsbox.IQRScaler",
                        "hyperparameters": {}
                    },
                    "d3m.primitives.dsbox.DoNothing",
                ],
                "inputs": ["impute_step"]
            },
            # {
            #     "name": "cast_1_step",  # turn columns to float
            #     # "name": data,
            #     "primitives": [
            #         {
            #             "primitive": "d3m.primitives.data.CastToType",
            #             "hyperparameters": 
            #                             {
            #                             "type_to_cast": ["float"],
            #                             "exclude_columns": (0,),
            #                             "use_columns": (0,),
            #                             }
            #         },
            #         "d3m.primitives.dsbox.DoNothing",
            #     ],
            #     # "inputs": ["scaler_step"]
            #     "inputs":["extract_attribute_step"]
            # },
            {
                "name": "PCA_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.sklearn_wrap.SKPCA",
                        "hyperparameters":
                        { 
                            'use_semantic_types':[True],
                            'return_result':['new'],
                            'add_index_columns':[True],
                            'n_components': [10, 15, 25]
                        }
                    },
                    "d3m.primitives.dsbox.DoNothing",
                ],
                # "inputs": ["cast_1_step"]
                "inputs":["scaler_step"]
            },
            {
                "name": data,
                "primitives": [
                    # {
                    #     "primitive": "d3m.primitives.data.CastToType",
                    #     "hyperparameters": {"type_to_cast": ["float"]}
                    # },
                    "d3m.primitives.dsbox.DoNothing",
                ],
                # "inputs": ["scaler_step"]
                "inputs":["PCA_step"]
            },
            {
                "name": target,
                "primitives": [{
                    "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                # 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["to_dataframe_step"]
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
                "primitives": [
                    "d3m.primitives.dsbox.CleaningFeaturizer",
                    "d3m.primitives.dsbox.DoNothing",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "corex_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.dsbox.CorexText",
                        "hyperparameters":
                            {
                                'n_hidden': [5, 10],
                                'threshold': [0, 500],
                                'n_grams': [1, 3],
                            }
                    },
                ],
                "inputs": ["clean_step"]
            },
            {
                "name": "encoder_step",
                "primitives": [
                    "d3m.primitives.dsbox.Encoder",
                    "d3m.primitives.dsbox.DoNothing"
                ],
                "inputs": ["corex_step"]
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                "inputs": ["encoder_step"]
            },
            {
                "name": "scaler_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.sklearn_wrap.SKMaxAbsScaler",
                        "hyperparameters": {}
                    },
                    {
                        "primitive": "d3m.primitives.dsbox.IQRScaler",
                        "hyperparameters": {}
                    },
                    "d3m.primitives.dsbox.DoNothing",
                ],
                "inputs": ["impute_step"]
            },
            {
                "name": data,
                "primitives": [
                    {
                        "primitive": "d3m.primitives.data.CastToType",
                        "hyperparameters": {"type_to_cast": ["float"]}
                    },
                    "d3m.primitives.dsbox.DoNothing",
                ],
                "inputs": ["scaler_step"]
            },
            {
                "name": target,
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
        ]

    @staticmethod
    def human_steps():
        '''
        simulated solution steps for classification and regression
        '''
        return [
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
                "primitives": [
                    "d3m.primitives.dsbox.CleaningFeaturizer",
                    "d3m.primitives.dsbox.DoNothing",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encoder_step",
                "primitives": [
                    "d3m.primitives.dsbox.Encoder",
                    "d3m.primitives.dsbox.Labler"
                ],
                "inputs": ["clean_step"]
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.dsbox.MeanImputation"],
                "inputs": ["encoder_step"]
            },
            {
                "name": "cast_step",  # turn columns to float
                "primitives": [
                    {
                        "primitive": "d3m.primitives.data.CastToType",
                        "hyperparameters": {"type_to_cast": ["float"]}
                    },
                    "d3m.primitives.dsbox.DoNothing",
                ],
                "inputs": ["impute_step"]
            },
            {
                "name": "extract_target_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
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
        ]

    @staticmethod
    def dsbox_feature_selector(ptype, first_input='cast_step', second_input='extract_target_step'):
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
                            "primitive": "d3m.primitives.sklearn_wrap.SKSelectFwe",
                            "hyperparameters": {
                            'use_semantic_types':[True],
                                'return_result':['new'],
                                'add_index_columns':[True],
                                "alpha": [float(x) for x in np.logspace(-4, -1, 6)]
                            }
                        },
                        {
                            "primitive": "d3m.primitives.sklearn_wrap.SKGenericUnivariateSelect",
                            "hyperparameters": {
                                'use_semantic_types':[True],
                                'return_result':['new'],
                                'add_index_columns':[True],
                                "score_func": ["f_regression"],
                                "mode": ["percentile"],
                                "param": [5, 7, 10, 15, 30, 50, 75],
                            }
                        },
                        "d3m.primitives.dsbox.DoNothing"
                    ],
                    "inputs":[first_input, second_input]
                },
            ]
        else:
            return [
                {
                    "name": "feature_selector_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.sklearn_wrap.SKSelectFwe",
                            "hyperparameters": {
                            'use_semantic_types':[True],
                                'return_result':['new'],
                                'add_index_columns':[True],
                                "alpha": [float(x) for x in np.logspace(-4, -1, 6)]
                            }
                        },

                        {
                            "primitive": "d3m.primitives.sklearn_wrap.SKGenericUnivariateSelect",
                            "hyperparameters": {
                                'use_semantic_types':[True],
                                'return_result':['new'],
                                'add_index_columns':[True],
                                "mode": ["percentile"],
                                "param": [5, 7, 10, 15, 30, 50, 75],
                            }
                        },
                        "d3m.primitives.dsbox.DoNothing"

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
                            "primitive": "d3m.primitives.dsbox.DoNothing",
                        },
                        {
                            "primitive": "d3m.primitives.sklearn_wrap.SKPCA",
                            "hyperparameters":
                                {'n_components': [(2), (4), (8), (16), (32), (64), (128)], }
                        },
                        {
                            "primitive": "d3m.primitives.sklearn_wrap.SKKernelPCA",
                            "hyperparameters":
                                {
                                    'n_components': [(2), (4), (8), (16), (32), (64), (128)],
                                    'kernel': [('rbf'), ('sigmoid'), ('cosine')]
                                }
                        },
                        {
                            "primitive": "d3m.primitives.sklearn_wrap.SKKernelPCA",
                            "hyperparameters":
                                {
                                    'n_components': [(2), (4), (8), (16), (32), (64), (128)],
                                    'kernel': [('poly')],
                                    'degree': [(2), (3), (4)],
                                }
                        },
                        {
                            "primitive": "d3m.primitives.sklearn_wrap.SKVarianceThreshold",
                            "hyperparameters":
                                {'threshold': [(0.01), (0.01), (0.05), (0.1)], }
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
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
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
                    "primitives": ["d3m.primitives.data.ColumnParser"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": attribute_name,
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
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": clean_name,
                    "primitives": [
                        "d3m.primitives.dsbox.CleaningFeaturizer",
                        "d3m.primitives.dsbox.DoNothing"
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
                        # {"primitive": "d3m.primitives.dsbox.DoNothing", },
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

    @staticmethod
    def dsbox_imputer(encoded_name: str = "cast_1_step",
                      impute_name: str = "impute_step"):
        return \
            [
                {
                    "name": "base_impute_step",
                    "primitives": [
                        {"primitive": "d3m.primitives.dsbox.MeanImputation", },
                        # {"primitive": "d3m.primitives.dsbox.GreedyImputation", },
                        {"primitive": "d3m.primitives.dsbox.MeanImputation", },
                        {"primitive": "d3m.primitives.dsbox.IterativeRegressionImputation", },
                        # {"primitive": "d3m.primitives.dsbox.DoNothing", },
                    ],
                    "inputs": [encoded_name]
                },
                {
                    "name": impute_name,
                    "primitives": [
                        {"primitive": "d3m.primitives.sklearn_wrap.SKMaxAbsScaler", },  # IQR always create error
                        {"primitive": "d3m.primitives.dsbox.DoNothing", },
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
                            "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        "hyperparameters":
                            {
                                'max_depth': [(2), (4), (8)],  # (10), #
                                'n_estimators': [(2), (5), (10), (20), (30), (40)]
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

    @staticmethod
    def regression_model(feature_name: str = "impute_step",
                         target_name: str = "extract_target_step"):
        return \
            [
                {
                    "name": "model_step",
                    "primitives": [
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

    @staticmethod
    def class_hyperparameter_generator(primitive_name, parameter_name, definition):
        from d3m import index
        g = None
        try:
            g = index.get_primitive(primitive_name).metadata.query()["primitive_code"]["hyperparams"][parameter_name]['structural_type'](definition)
        except:
            print("[ERROR] Hyperparameter not valid!")
            pass
        return g