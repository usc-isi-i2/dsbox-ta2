import json
import glob
import typing
import numpy as np

from d3m import index
from d3m.container.dataset import SEMANTIC_TYPES
from d3m.metadata.problem import TaskType, TaskSubtype
from d3m.container.list import List
from dsbox.template.template import TemplatePipeline, DSBoxTemplate
from dsbox.template.dsbox_templates.template_steps import TemplateSteps
import copy


from dsbox.template.dsbox_templates.dsbox_classification import *

from dsbox.template.dsbox_templates.dsbox_regression import *

from dsbox.template.dsbox_templates.dsbox_large_datasets import *

from dsbox.template.dsbox_templates.dsbox_timeseries_forcasting import *

from dsbox.template.dsbox_templates.dsbox_timeseries_general import *

from dsbox.template.dsbox_templates.dsbox_image import *

from dsbox.template.dsbox_templates.dsbox_text import *

from dsbox.template.dsbox_templates.dsbox_graph import *

from dsbox.template.dsbox_templates.dsbox_TA1 import *

from dsbox.template.dsbox_templates.dsbox_other import *


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
            "default_classification_template": DefaultClassificationTemplate,
            "default_regression_template": DefaultRegressionTemplate,
            "classification_with_feature_selection":ClassificationWithSelection,
            "regression_with_feature_selection":RegressionWithSelection,

            "Large_column_number_with_numerical_only_classification":Large_column_number_with_numerical_only_classification,
            "Large_column_number_with_numerical_only_regression":Large_column_number_with_numerical_only_regression,

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

            # older templates
            "dsbox_classification_template": dsboxClassificationTemplate,
            "dsbox_regression_template": dsboxRegressionTemplate,
            "CMU_Clustering_Template": CMUClusteringTemplate,
            "Default_timeseries_collection_template": DefaultTimeseriesCollectionTemplate,
            "Default_image_processing_regression_template":
                DefaultImageProcessingRegressionTemplate,
            "TA1DefaultImageProcessingRegressionTemplate":
                TA1DefaultImageProcessingRegressionTemplate,
            "TA1_classification_template_1": TA1ClassificationTemplate1,
            "MuxinTA1ClassificationTemplate1": MuxinTA1ClassificationTemplate1,
            "MuxinTA1ClassificationTemplate2": MuxinTA1ClassificationTemplate2,
            "UU3_Test_Template": UU3TestTemplate,
            "TA1Classification_2": TA1Classification_2,
            "TA1Classification_3": TA1Classification_3,
            "TA1VggImageProcessingRegressionTemplate": TA1VggImageProcessingRegressionTemplate,
            "SRI_LinkPrediction_Template": SRILinkPredictionTemplate,

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
            "Michigan_Video_Classification_Template": MichiganVideoClassificationTemplate,
            "DefaultTimeseriesRegressionTemplate": DefaultTimeseriesRegressionTemplate,
            "TimeSeriesForcastingTestingTemplate": TimeSeriesForcastingTestingTemplate,
            "TemporaryObjectDetectionTemplate": TemporaryObjectDetectionTemplate
        }

        if run_single_template:
            self._load_single_inline_templates(run_single_template)
        else:
            self._load_inline_templates()

    def get_templates(self, task: TaskType, subtype: TaskSubtype, taskSourceType: SEMANTIC_TYPES) \
            -> typing.List[DSBoxTemplate]:
        results = []
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
        # template that gives us the mean baseline as a result
        self.templates.append(SRIMeanBaselineTemplate)

        self.templates.append(DefaultTimeseriesRegressionTemplate)

        # default tabular templates, encompassing many of the templates below
        self.templates.append(DefaultClassificationTemplate)
        self.templates.append(NaiveBayesClassificationTemplate)

        self.templates.append(DefaultRegressionTemplate)

        # new tabular classification
        # self.templates.append(RandomForestClassificationTemplate)
        # self.templates.append(ExtraTreesClassificationTemplate)
        # self.templates.append(GradientBoostingClassificationTemplate)
        # takes too long to run self.templates.append(SVCClassificationTemplate)

        # new tabular regression
        # self.templates.append(RandomForestRegressionTemplate)
        # self.templates.append(ExtraTreesRegressionTemplate)
        # self.templates.append(GradientBoostingRegressionTemplate)
        # takes too long to run self.templates.append(SVRRegressionTemplate)

        # text templates, but also support tabular data
        self.templates.append(DefaultTextClassificationTemplate)
        self.templates.append(DefaultTextRegressionTemplate)

        # Tabular Classification
        # self.templates.append(TA1Classification_3)
        # self.templates.append(MuxinTA1ClassificationTemplate1)
        self.templates.append(UU3TestTemplate)
        # self.templates.append(TA1ClassificationTemplate1)

        # Image Regression
        self.templates.append(DefaultImageProcessingRegressionTemplate)

        # Others
        self.templates.append(DefaultTimeseriesCollectionTemplate)
        self.templates.append(TimeSeriesForcastingTestingTemplate)

        self.templates.append(SRILinkPredictionTemplate)
        self.templates.append(SRICommunityDetectionTemplate)
        self.templates.append(SRIGraphMatchingTemplate)
        self.templates.append(SRIVertexNominationTemplate)

        self.templates.append(BBNAudioClassificationTemplate)
        self.templates.append(SRICollaborativeFilteringTemplate)
        self.templates.append(DefaultTimeSeriesForcastingTemplate)
        self.templates.append(CMUClusteringTemplate)
        self.templates.append(MichiganVideoClassificationTemplate)
        self.templates.append(TemporaryObjectDetectionTemplate) 
        # self.templates.append(JHUVertexNominationTemplate)
        # self.templates.append(JHUGraphMatchingTemplate)

        # templates used for datasets with a large number of columns
        self.templates.append(Large_column_number_with_numerical_only_classification)
        self.templates.append(Large_column_number_with_numerical_only_regression)
        
        self.templates.append(ClassificationWithSelection)
        self.templates.append(RegressionWithSelection)

        # dsbox all in one templates
        # move dsboxClassificationTemplate to last execution because sometimes this template have bugs
        self.templates.append(dsboxClassificationTemplate)
        self.templates.append(dsboxRegressionTemplate)

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






################################################################################################################




######################################            Templates            #########################################




################################################################################################################


