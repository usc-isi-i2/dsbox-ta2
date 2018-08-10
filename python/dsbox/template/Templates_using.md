# **default_classification_template**

| TemplateSteps                              |                                                                                                                Primitives |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                                                         dsbox.Denormalize |
| to_dataframe_step                          |                                                                                                        DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                                                             ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                                                            dsbox.Profiler |
| clean_step                                 |                                                                                                  dsbox.CleaningFeaturizer |
| corex_step                                 |                                                                                                           dsbox.CorexText |
| encoder_step                               |                                                                                            dsbox.Encoder, dsbox.DoNothing |
| impute_step                                |                                                                                                      dsbox.MeanImputation |
| scaler_step                                |                                                             sklearn_wrap.SKMaxAbsScaler, dsbox.IQRScaler, dsbox.DoNothing |
| cast_to_type_step                          |                                                                                               CastToType, dsbox.DoNothing |
| dimension_reduction_step                   |                                                                                       sklearn_wrap.SKPCA, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKRandomForestClassifier,   sklearn_wrap.SKExtraTreesClassifier,   sklearn_wrap.SKGradientBoostingClassifier |
