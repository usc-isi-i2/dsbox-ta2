import logging

import numpy as np

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
                "name": "to_dataframe_step",
                "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                "inputs": ["template_input"]
            },
            {
                "name": "extract_attribute_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.CommOn",
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
                    # "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encode_step",
                "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                "inputs": ["clean_step"]
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
                "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                "inputs": ["to_numeric_step"]
            },
            {
                "name": "scaler_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn",
                        "hyperparameters":
                        {
                            'use_semantic_types':[True],
                            'return_result':['new'],
                            'add_index_columns':[True],
                        }
                    },
                    {
                        "primitive": "d3m.primitives.normalization.iqr_scaler.DSBOX",
                        "hyperparameters": {}
                    },
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["impute_step"]
            },
            {
                "name": data,
                "primitives": [
                    {
                        "primitive": "d3m.primitives.feature_extraction.pca.SKlearn",
                        "hyperparameters":
                        {
                            'use_semantic_types': [True],
                            'add_index_columns': [True],
                            'return_result': ['new'],
                            'n_components': [10, 15, 25]
                        }
                    },
                "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["scaler_step"]
            },
            {
                "name": "pre_"+target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.CommOn",
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
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["pre_"+target]
            }
        ]

    @staticmethod
    def dsbox_augmentation_step(datamart_search_results, large_dataset=False):
        '''
        dsbox generic step for classification and regression, directly lead to model step
        '''
        wikidata_search_results = []
        vector_search_results = []
        general_search_results = []
        for each_result in datamart_search_results:
            detail_info = each_result.get_json_metadata()
            # if it is wikidata search reuslts
            if detail_info['summary']['Datamart ID'].startswith("wikidata_search"):
                wikidata_search_results.append(each_result)
            elif detail_info['summary']['Datamart ID'].startswith("vector_search"):
                vector_search_results.append(each_result)
            else:
                general_search_results.append(each_result)
        all_steps = []
        augment_step_number = 0

        if not large_dataset:
            res1, augment_step_number = TemplateSteps.add_steps_serial(wikidata_search_results, augment_step_number)
            all_steps.extend(res1)
            res2, augment_step_number = TemplateSteps.add_steps_serial(vector_search_results, augment_step_number)
            all_steps.extend(res2)

        res3, augment_step_number = TemplateSteps.add_steps_parallel(general_search_results, augment_step_number)
        all_steps.extend(res3)

        # remove all q nodes here, otherwise cleaner may generate a lot of useless columns
        if len(all_steps) > 0:
            to_dataframe_step = {
                "name": "to_dataframe_step",
                "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                "inputs": ["augment_step" + str(augment_step_number - 1)]
            }
            all_steps.append(to_dataframe_step)

            from datamart_isi.config import q_node_semantic_type
            remove_q_nodes_step = {
                    "name": "remove_q_nodes_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_transformation.remove_semantic_types.DataFrameCommon",
                            "hyperparameters":
                            {
                                'semantic_types':[(q_node_semantic_type,)],
                            }
                        }
                    ],
                    "inputs": ["to_dataframe_step"]
                }
            all_steps.append(remove_q_nodes_step)

        return all_steps


    @staticmethod
    def add_steps_serial(search_results, start_step):
        """
            Add all search results in serial steps as candidates
        """
        augment_steps = []
        for i, each in enumerate(search_results, start_step):
            each_augment_step = {
                "name": "augment_step" + str(i),
                "primitives": [
                    "d3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX", 
                    {
                        "primitive": "d3m.primitives.data_augmentation.datamart_augmentation.Common",
                        "hyperparameters":
                        {
                            'system_identifier':["ISI"],
                            'search_result':[each.serialize()],
                        }
                    }
                ],
                "inputs": ["template_input" if i==0 else "augment_step" + str(i - 1)]
            }
            augment_steps.append(each_augment_step)
        
        return augment_steps, start_step + len(search_results)


    @staticmethod
    def add_steps_parallel(search_results, start_step):
        """
            Add all search results in one step as candidates
        """
        augment_steps = []

        search_result = []
        for each in search_results:
            search_result.append(each.serialize())

        each_augment_step = {
            "name": "augment_step" + str(start_step),
            "primitives": [
                "d3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX", 
                {
                    "primitive": "d3m.primitives.data_augmentation.datamart_augmentation.Common",
                    "hyperparameters":
                    {
                        'system_identifier':["ISI"],
                        'search_result':search_result,
                    }
                }
            ],
            "inputs": ["template_input" if start_step==0 else "augment_step" + str(start_step - 1)]
        }
        augment_steps.append(each_augment_step)
        
        return augment_steps, start_step + 1


    # Returns a list of dicts with the most common steps
    @staticmethod
    def dsbox_generic_text_steps(data: str = "data", target: str = "target"):
        '''
        dsbox generic step for classification and regression, specify on text data processing, directly lead to model step
        '''
        return [
            {
                "name": "to_dataframe_step",
                "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                "inputs": ["template_input"]
            },
            {
                "name": "extract_attribute_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.CommOn",
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
                "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encoder_step",
                "primitives": [
                    "d3m.primitives.data_preprocessing.encoder.DSBOX",
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX"
                ],
                "inputs": ["clean_step"]
            },
            {
                "name": "corex_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.feature_construction.corex_text.DSBOX",
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
                "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                "inputs":["corex_step"],
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                "inputs": ["to_numeric_step"]
            },
            {
                "name": data,
                "primitives": [
                    {
                        "primitive": "d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn",
                        "hyperparameters": {
                            'add_index_columns': [True],
                            'use_semantic_types': [True],
                        }
                    },
                    {
                        "primitive": "d3m.primitives.normalization.iqr_scaler.DSBOX",
                        "hyperparameters": {}
                    },
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["impute_step"]
            },
            {
                "name": "pre_"+target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.CommOn",
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
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
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
                "name": "to_dataframe_step",
                "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                "inputs": ["template_input"]
            },
            {
                "name": "extract_attribute_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.CommOn",
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
                "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encoder_step",
                "primitives": [
                    "d3m.primitives.data_preprocessing.encoder.DSBOX",
                    "d3m.primitives.data_cleaning.label_encoder.DSBOX"
                ],
                "inputs": ["clean_step"]
            },
            {
                "name": "to_numeric_step",
                "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                "inputs":["encoder_step"],
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                "inputs": ["to_numeric_step"]
            },
            {
                "name": "pre_extract_target_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.CommOn",
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
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["pre_extract_target_step"]
            },
        ]

    @staticmethod
    def dsbox_feature_selector(ptype, first_input='impute_step', second_input='extract_target_step'):
        '''
        dsbox feature selection steps for classification and regression, lead to feature selector steps
        '''
        if ptype == "regression":
            return [
                {
                    "name": "feature_selector_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_selection.select_fwe.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                "alpha": [float(x) for x in np.logspace(-4, -1, 6)]
                            }
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
                        {
                            "primitive": "d3m.primitives.feature_selection.joint_mutual_information.AutoRPI",
                            "hyperparameters": {
                                #'method': ["counting", "pseudoBayesian", "fullBayesian"],
                                'nbins': [2, 5, 10, 13, 20]
                                }
                        },
                        "d3m.primitives.data_preprocessing.do_nothing.DSBOX"
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
                            "primitive": "d3m.primitives.feature_selection.select_fwe.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                "alpha": [float(x) for x in np.logspace(-4, -1, 6)]
                            }
                        },
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
                        {
                            "primitive": "d3m.primitives.feature_selection.joint_mutual_information.AutoRPI",
                            "hyperparameters": {
                                #'method': ["counting", "pseudoBayesian", "fullBayesian"],
                                'nbins': [2, 5, 10, 13, 20]
                                }
                        },
                        "d3m.primitives.feature_selection.simultaneous_markov_blanket.AutoRPI",
                        "d3m.primitives.data_preprocessing.do_nothing.DSBOX"

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
                            "primitive": "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                        },
                        {
                            "primitive": "d3m.primitives.feature_extraction.pca.SKlearn",
                            "hyperparameters":
                                {
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                    'n_components': [(2), (4), (8), (16), (32), (64), (128)], }
                        },
                        {
                            "primitive": "d3m.primitives.feature_extraction.kernel_pca.SKlearn",
                            "hyperparameters":
                                {
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                    'n_components': [(2), (4), (8), (16), (32), (64), (128)],
                                    'kernel': [('rbf'), ('sigmoid'), ('cosine')]
                                }
                        },
                        {
                            "primitive": "d3m.primitives.feature_extraction.kernel_pca.SKlearn",
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
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": attribute_name,
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.CommOn",
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
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.CommOn",
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
                        "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
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
                            "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                            "hyperparameters": {}
                        },
                        {
                            "primitive": "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
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
                    "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": clean_name,
                    "primitives": [
                        "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                        "d3m.primitives.data_preprocessing.do_nothing.DSBOX"
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
                            "primitive": "d3m.primitives.feature_construction.corex_text.DSBOX",
                            "hyperparameters":
                                {
                                    'n_hidden': [(10)],
                                    'threshold': [(0), (500)],
                                    'n_grams': [(1), (5)],
                                    'max_df': [(.9)],
                                    'min_df': [(.02)],
                                }
                        },
                        {"primitive": "d3m.primitives.data_preprocessing.do_nothing.DSBOX", },
                    ],
                    "inputs": [clean_name]
                },
                # {
                #     "name": "encode_unary_step",
                #     "primitives": [
                #         {"primitive": "d3m.primitives.data_preprocessing.unary_encoder.DSBOX", },
                #         {"primitive": "d3m.primitives.data_preprocessing.do_nothing.DSBOX", },
                #     ],
                #     "inputs": ["encode_text_step"]
                # },
                {
                    # "name": 'encode_string_step',
                    "name": encoded_name,
                    "primitives": [
                        {"primitive": "d3m.primitives.data_preprocessing.encoder.DSBOX", },
                        {"primitive": "d3m.primitives.data_cleaning.label_encoder.DSBOX", },
                        # {"primitive": "d3m.primitives.data_preprocessing.do_nothing.DSBOX", },
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
                        {"primitive": "d3m.primitives.data_preprocessing.mean_imputation.DSBOX", },
                        {"primitive": "d3m.primitives.data_preprocessing.low_rank_imputer.Cornell", },
                        {"primitive": "d3m.primitives.data_preprocessing.mean_imputation.DSBOX", },
                        {"primitive": "d3m.primitives.data_preprocessing.iterative_regression_imputation.DSBOX", },
                        # {"primitive": "d3m.primitives.data_preprocessing.do_nothing.DSBOX", },
                    ],
                    "inputs": [encoded_name]
                },
                {
                    "name": impute_name,
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                            }
                        },
                        {
                            "primitive": "d3m.primitives.normalization.iqr_scaler.DSBOX",
                            "hyperparameters": {
                                "use_semantic_types": [True],
                                "add_index_columns": [True],
                            }
                        },
                        {"primitive": "d3m.primitives.data_preprocessing.do_nothing.DSBOX", },
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
                    },{
                            "primitive":
                                "d3m.primitives.classification.xgboost_gbtree.DataFrameCommon",
                            "hyperparameters":
                                {
                                    'use_semantic_types': [True],
                                    'add_index_columns': [True]
                                }
                    }
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
