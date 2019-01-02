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



## **default_regression_template**

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
| model_step                                 | sklearn_wrap.SKGradientBoostingRegressor,   sklearn_wrap.SKExtraTreesRegressor,   sklearn_wrap.SKRandomForestRegressor |


## **svr_regression_template**

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
| model_step                                 | sklearn_wrap.SKSVR |


## **svr_regression_template**

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
| model_step                                 | sklearn_wrap.SKGradientBoostingRegressor(with more tuning parameters) |



## **random_forest_regression_template**

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
| model_step                                 | sklearn_wrap.SKRandomForestRegressor(with more tuning parameters) |


## **regression_with_feature_selection**


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
| model_step                                 |           sklearn_wrap.SKSGDRegressor,   sklearn_wrap.SKGradientBoostingRegressor |



## **dsbox_regressoin_template**

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
| model_step                                 | sklearn_wrap.SKRidge, sklearn_wrap.SKLars,   d3m.primitives.sklearn_wrap.SKKNeighborsRegressor, d3m.primitives.sklearn_wrap.SKLinearSVR, d3m.primitives.sklearn_wrap.SKSGDRegressor, d3m.primitives.sklearn_wrap.SKGradientBoostingRegressor|


## **Large_column_number_with_numerical_only_classification**


| TemplateSteps                              |                            Primitives |
|:--------------------------------------------|--------------------------------------:|
| denormalize_step                           |                     dsbox.Denormalize |
| to_dataframe_step                          |                    DatasetToDataFrame |
| extract_attribute_step/extract_target_step |         ExtractColumnsBySemanticTypes |
| encoder_step                               |                          dsbox.Labler |
| cast_to_type_step                          |           CastToType, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKRandomForestClassifier |


## **Large_column_number_with_numerical_only_regression**


| TemplateSteps                              |                            Primitives |
|:--------------------------------------------|--------------------------------------:|
| denormalize_step                           |                     dsbox.Denormalize |
| to_dataframe_step                          |                    DatasetToDataFrame |
| extract_attribute_step/extract_target_step |         ExtractColumnsBySemanticTypes |
| encoder_step                               |                          dsbox.Labler |
| cast_to_type_step                          |           CastToType, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKRandomForestRegressor |



## **Default_Time_Series_Forcasting_Template**


| TemplateSteps                              |                                    Primitives |
|:--------------------------------------------|----------------------------------------------:|
| denormalize_step                           |                             dsbox.Denormalize |
| to_dataframe_step                          |                            DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                 ExtractColumnsBySemanticTypes |
| profiler_step                              |                                dsbox.Profiler |
| clean_step                                 |                      dsbox.CleaningFeaturizer |
| timeseries_to_list_step                    |                        dsbox.TimeseriesToList |
| random_projection_step                     | dsbox.RandomProjectionTimeSeriesFeaturization |
| cast_to_type_step                          |                   CastToType, dsbox.DoNothing |
| model_step                                 |          sklearn_wrap.SKRandomForestRegressor |


## **Default_timeseries_collection_template**


| TemplateSteps                              |                                    Primitives |
|:--------------------------------------------|----------------------------------------------:|
| denormalize_step                           |                             dsbox.Denormalize |
| to_dataframe_step                          |                            DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                 ExtractColumnsBySemanticTypes |
| timeseries_to_list_step                    |                        dsbox.TimeseriesToList |
| random_projection_step                     | dsbox.RandomProjectionTimeSeriesFeaturization |
| model_step                                 |         sklearn_wrap.SKRandomForestClassifier |


## **Default_timeseries_regression_template**

| TemplateSteps                              |                                    Primitives |
|:--------------------------------------------|----------------------------------------------:|
| denormalize_step                           |                             dsbox.Denormalize |
| to_dataframe_step                          |                            DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                 ExtractColumnsBySemanticTypes |
| profiler_step                              |                                dsbox.Profiler |
| clean_step                                 |                      dsbox.CleaningFeaturizer |
| timeseries_to_list_step                    |                        dsbox.TimeseriesToList |
| random_projection_step                     | dsbox.RandomProjectionTimeSeriesFeaturization |
| cast_to_type_step                          |                   CastToType, dsbox.DoNothing |
| model_step                                 |          sklearn_wrap.SKRandomForestRegressor |


## **TemporaryObjectDetectionTemplate**

| TemplateSteps                              |                           Primitives |
|:--------------------------------------------|-------------------------------------:|
| denormalize_step                           |                    dsbox.Denormalize |
| to_dataframe_step                          |                   DatasetToDataFrame |
| extract_attribute_step/extract_target_step |        ExtractColumnsBySemanticTypes |
| to_tensor_step                             |              dsbox.DataFrameToTensor |
| image_processing_step                      |           dsbox.ResNet50ImageFeature |
| data_clean_step                            |             dsbox.CleaningFeaturizer |
| model_step                                 | sklearn_wrap.SKRandomForestRegressor |


## **Default_image_processing_regression_template**

| TemplateSteps                              |                           Primitives |
|:--------------------------------------------|-------------------------------------:|
| denormalize_step                           |                    dsbox.Denormalize |
| to_dataframe_step                          |                   DatasetToDataFrame |
| extract_attribute_step/extract_target_step |        ExtractColumnsBySemanticTypes |
| feature_extraction_step                    |           dsbox.ResNet50ImageFeature |
| PCA_step                                   |                   sklearn_wrap.SKPCA |
| model_step                                 | sklearn_wrap.SKRandomForestRegressor |



## **default_text_classification_template**


| TemplateSteps                              |                                                                                                       Primitives |
|:--------------------------------------------|-----------------------------------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                                                dsbox.Denormalize |
| to_dataframe_step                          |                                                                                               DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                                                    ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                                                   dsbox.Profiler |
| clean_step                                 |                                                                                         dsbox.CleaningFeaturizer |
| corex_step                                 |                                                                                                  dsbox.CorexText |
| encoder_step                               |                                                                                   dsbox.Encoder, dsbox.DoNothing |
| impute_step                                |                                                                                             dsbox.MeanImputation |
| scaler_step                                |                                                    sklearn_wrap.SKMaxAbsScaler, dsbox.IQRScaler, dsbox.DoNothing |
| cast_to_type_step                          |                                                                                      CastToType, dsbox.DoNothing |
| dimension_reduction_step                   |                                                                              sklearn_wrap.SKPCA, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKGradientBoostingClassifier, sklearn_wrap.SKMultinomialNB,   sklearn_wrap.SKRandomForestClassifier |



## **default_text_regression_template**

| TemplateSteps                              |                                                                                                       Primitives |
|:--------------------------------------------|-----------------------------------------------------------------------------------------------------------------:|
| denormalize_step                           |                                                                                                dsbox.Denormalize |
| to_dataframe_step                          |                                                                                               DatasetToDataFrame |
| extract_attribute_step/extract_target_step |                                                                                    ExtractColumnsBySemanticTypes |
| profiler_step                              |                                                                                                   dsbox.Profiler |
| clean_step                                 |                                                                                         dsbox.CleaningFeaturizer |
| corex_step                                 |                                                                                                  dsbox.CorexText |
| encoder_step                               |                                                                                   dsbox.Encoder, dsbox.DoNothing |
| impute_step                                |                                                                                             dsbox.MeanImputation |
| scaler_step                                |                                                    sklearn_wrap.SKMaxAbsScaler, dsbox.IQRScaler, dsbox.DoNothing |
| cast_to_type_step                          |                                                                                      CastToType, dsbox.DoNothing |
| dimension_reduction_step                   |                                                                              sklearn_wrap.SKPCA, dsbox.DoNothing |
| model_step                                 | sklearn_wrap.SKRandomForestRegressor, sklearn_wrap.SKGradientBoostingRegressor  |


## **LinkPrediction_GraphMatching_VertexNoimination_Test_Template**


| TemplateSteps                              |                            Primitives |
|:--------------------------------------------|--------------------------------------:|
| denormalize_step                           |                           Denormalize |
| to_dataframe_step                          |                    DatasetToDataFrame |
| extract_attribute_step/extract_target_step |         ExtractColumnsBySemanticTypes |
| model_step                                 | sklearn_wrap.SKRandomForestClassifier |


## **SRI_GraphMatching_Template**

| TemplateSteps |                          Primitives |
|:---------------|------------------------------------:|
| model_step    | sri.psl.GraphMatchingLinkPrediction |

## **SRI_Vertex_Nomination_Template**

| TemplateSteps |                       Primitives |
|:---------------|---------------------------------:|
| parse_step    | sri.graph.VertexNominationParser |
| model_step    |         sri.psl.VertexNomination |


## **SRI_Community_Detection_Template**


| TemplateSteps |                         Primitives |
|:---------------|-----------------------------------:|
| parse_step    | sri.graph.CommunityDetectionParser |
| model_step    |         sri.psl.CommunityDetection |


## **SRI_Collaborative_Filtering_Template**


| TemplateSteps |                                   Primitives |
|:---------------|---------------------------------------------:|
| model_step    | sri.psl.CollaborativeFilteringLinkPrediction |



## **BBN_Audio_Classification_Template**

| TemplateSteps        |                        Primitives |
|:----------------------|----------------------------------:|
| denormalize_step     |                       Denormalize |
| to_dataframe_step    |                DatasetToDataFrame |
| extract_target_step  |     ExtractColumnsBySemanticTypes |
| readaudio_step       |       bbn.time_series.AudioReader |
| channel_step         |   bbn.time_series.ChannelAverager |
| signaldither_step    |      bbn.time_series.SignalDither |
| signalframer_step    |      bbn.time_series.SignalFramer |
| MFCC_step            |        bbn.time_series.SignalMFCC |
| vectorextractor_step |  bbn.time_series.IVectorExtractor |
| model_step           | bbn.sklearn_wrap.BBNMLPClassifier |


## **Michigan_Video_Classification_Template**

| TemplateSteps       |                    Primitives |
|:---------------------|------------------------------:|
| denormalize_step    |                   Denormalize |
| to_dataframe_step   |            DatasetToDataFrame |
| extract_target_step | ExtractColumnsBySemanticTypes |
| read_video_step     |                   VideoReader |
| featurize_step      |      spider.featurization.I3D |
| convert_step        |            NDArrayToDataFrame |
| model_step          |                  RandomForest |


## **CMU_Clustering_Template**


| TemplateSteps          |                    Primitives |
|:------------------------|------------------------------:|
| denormalize_step       |                   Denormalize |
| to_dataframe_step      |            DatasetToDataFrame |
| extract_attribute_step | ExtractColumnsBySemanticTypes |
| model_step             |               cmu.fastlvm.GMM |


## **SRI_Mean_Baseline_Template**

| TemplateSteps |                Primitives |
|:---------------|--------------------------:|
| model_step    | sri.baseline.MeanBaseline |