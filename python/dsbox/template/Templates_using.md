# Templates, Steps and Primitives

## **default_classification_template**

| TemplateSteps                              |                                                                                                                Primitives |
|:--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------:|
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


## **naive_bayes_classification_template**

| TemplateSteps                              |                                                                            Primitives |
|:--------------------------------------------|--------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                     dsbox.Denormalize |
| to_dataframe_step                          |                                                                    DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                         ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                        dsbox.Profiler |
| clean_step                                 |                                                              dsbox.CleaningFeaturizer |
| corex_step                                 |                                                                       dsbox.CorexText |
| encoder_step                               |                                                        dsbox.Encoder, dsbox.DoNothing |
| impute_step                                |                                                                  dsbox.MeanImputation |
| scaler_step                                |                         sklearn_wrap.SKMaxAbsScaler, dsbox.IQRScaler, dsbox.DoNothing |
| cast_to_type_step                          |                                                           CastToType, dsbox.DoNothing |
| dimension_reduction_step                   |                                                   sklearn_wrap.SKPCA, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKBernoulliNB, sklearn_wrap.SKGaussianNB,   sklearn_wrap.SKMultinomialNB |


## **random_forest_classification_template**

| TemplateSteps                              |                                                                            Primitives |
|:--------------------------------------------|--------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                     dsbox.Denormalize |
| to_dataframe_step                          |                                                                    DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                         ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                        dsbox.Profiler |
| clean_step                                 |                                                              dsbox.CleaningFeaturizer |
| corex_step                                 |                                                                       dsbox.CorexText |
| encoder_step                               |                                                        dsbox.Encoder, dsbox.DoNothing |
| impute_step                                |                                                                  dsbox.MeanImputation |
| scaler_step                                |                         sklearn_wrap.SKMaxAbsScaler, dsbox.IQRScaler, dsbox.DoNothing |
| cast_to_type_step                          |                                                           CastToType, dsbox.DoNothing |
| dimension_reduction_step                   |                                                   sklearn_wrap.SKPCA, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKRandomForestClassifier(with more tuning parameters)|


## **extra_trees_classification_template**

| TemplateSteps                              |                                                                            Primitives |
|:--------------------------------------------|--------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                     dsbox.Denormalize |
| to_dataframe_step                          |                                                                    DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                         ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                        dsbox.Profiler |
| clean_step                                 |                                                              dsbox.CleaningFeaturizer |
| corex_step                                 |                                                                       dsbox.CorexText |
| encoder_step                               |                                                        dsbox.Encoder, dsbox.DoNothing |
| impute_step                                |                                                                  dsbox.MeanImputation |
| scaler_step                                |                         sklearn_wrap.SKMaxAbsScaler, dsbox.IQRScaler, dsbox.DoNothing |
| cast_to_type_step                          |                                                           CastToType, dsbox.DoNothing |
| dimension_reduction_step                   |                                                   sklearn_wrap.SKPCA, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKExtraTreesClassifier(with more tuning parameters)|


## **gradient_boosting_classification_template**

| TemplateSteps                              |                                                                            Primitives |
|:--------------------------------------------|--------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                     dsbox.Denormalize |
| to_dataframe_step                          |                                                                    DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                         ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                        dsbox.Profiler |
| clean_step                                 |                                                              dsbox.CleaningFeaturizer |
| corex_step                                 |                                                                       dsbox.CorexText |
| encoder_step                               |                                                        dsbox.Encoder, dsbox.DoNothing |
| impute_step                                |                                                                  dsbox.MeanImputation |
| scaler_step                                |                         sklearn_wrap.SKMaxAbsScaler, dsbox.IQRScaler, dsbox.DoNothing |
| cast_to_type_step                          |                                                           CastToType, dsbox.DoNothing |
| dimension_reduction_step                   |                                                   sklearn_wrap.SKPCA, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKGradientBoostingClassifier(with more tuning parameters)|


## **svc_classification_template**

| TemplateSteps                              |                                                                            Primitives |
|:--------------------------------------------|--------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                     dsbox.Denormalize |
| to_dataframe_step                          |                                                                    DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                         ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                        dsbox.Profiler |
| clean_step                                 |                                                              dsbox.CleaningFeaturizer |
| corex_step                                 |                                                                       dsbox.CorexText |
| encoder_step                               |                                                        dsbox.Encoder, dsbox.DoNothing |
| impute_step                                |                                                                  dsbox.MeanImputation |
| scaler_step                                |                         sklearn_wrap.SKMaxAbsScaler, dsbox.IQRScaler, dsbox.DoNothing |
| cast_to_type_step                          |                                                           CastToType, dsbox.DoNothing |
| dimension_reduction_step                   |                                                   sklearn_wrap.SKPCA, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKSVC(with more tuning parameters)|


## **classification_with_feature_selection**


| TemplateSteps                              |                                                                          Primitives |
|:--------------------------------------------|------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                   dsbox.Denormalize |
| to_dataframe_step                          |                                                                  DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                       ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                      dsbox.Profiler |
| clean_step                                 |                                                            dsbox.CleaningFeaturizer |
| encoder_step                               |                                                         dsbox.Encoder, dsbox.Labler |
| impute_step                                |                                                                dsbox.MeanImputation |
| cast_to_type_step                          |                                                         CastToType, dsbox.DoNothing |
| feature_selector_step                      | sklearn_wrap.SKSelectFwe,   sklearn_wrap.SKGenericUnivariateSelect, dsbox.DoNothing |
| model_step                                 |           sklearn_wrap.SKSGDClassifier,   sklearn_wrap.SKGradientBoostingClassifier |


## **dsbox_classification_template**

| TemplateSteps                              |                                                                                      Primitives |
|:--------------------------------------------|------------------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                               dsbox.Denormalize |
| to_dataframe_step                          |                                                                              DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                                   ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                                  dsbox.Profiler |
| clean_step                                 |                                                                        dsbox.CleaningFeaturizer |
| encode_text_step                           |                                                               dsbox.CorexText, dsbox .DoNothing |
| encode_other_step                          |                                                                   dsbox.Encoder,   dsbox.Labler |
| impute_step                                |                                        dsbox.MeanImputation,dsbox.IterativeRegressionImputation |
| scaler_step                                |                                                  sklearn_wrap.SKMaxAbsScaler,   dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKRandomForestClassifier, sklearn_wrap.SKLinearSVC,   sklearn_wrap.SKMultinomialNB |