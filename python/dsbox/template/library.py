import copy
import logging
import sys
import numpy as np  # type: ignore
import typing

from d3m import index
from d3m.metadata.problem import TaskKeyword
from dsbox.template.template_files.loaded import *
from dsbox.schema import SpecializedProblem
from dsbox.template.template import DSBoxTemplate
from dsbox.template.template_files.loaded.SKDummyTemplate import SKDummyTemplate

_logger = logging.getLogger(__name__)


"""
Attention!!!!!!
For anyone who add new templates here:
DO NOT ADD denomalize step from the original pipeline, 
(except problem related to graph, audio, time_series)
we will do denormalize at initilaization step
so everything passed from template_input is the Datasets already denormalized!

Updated v2019.11.14:
Now no more TaskType and TaskSubtype, but only TaskKeyword:
Current (from d3m v2019.11.10), following keywords are supported
{'classification','regression','clustering','linkPrediction','vertexNomination','vertexClassification','communityDetection','graphMatching','forecasting','collaborativeFiltering','objectDetection','semiSupervised','binary','multiClass','multiLabel','univariate','multivariate','overlapping','nonOverlapping','tabular','relational','image','audio','video','speech','text','graph','multiGraph','timeSeries','grouped','geospatial','remoteSensing','lupi','missingMetadata'}
"""

def have_intersection(lst1, lst2):
    if isinstance(lst1, str):
        lst1 = [lst1]
    if isinstance(lst2, str):
        lst2 = [lst2]
    return list(set(lst1) & set(lst2)) != []

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

        self.all_templates = {}

        if run_single_template:
            self._load_single_inline_templates(run_single_template)
        else:
            # pass
            self._load_inline_templates()


    def get_templates(self, task: typing.List[TaskKeyword], subtype: typing.List[TaskKeyword], taskSourceType: typing.Set,
                      specialized_problem: SpecializedProblem = SpecializedProblem.NONE) -> typing.List[DSBoxTemplate]:

        if type(task) is list:
            task = [x.name for x in task]
        else:
            task = [task.name]
        if type(subtype) is list:
            subtype = [x.name for x in subtype]
        else:
            subtype = [subtype.name]

        _logger.debug(f'Finding templates for Task={str(task)} and subtype={str(subtype)} resource={taskSourceType} special={specialized_problem}')
        results: typing.List[DSBoxTemplate] = []
        # 2019.7.18: temporary hacking here: only run special template for acled like problem
        # if specialized_problem == "Acled_problem":
        #     results = [CMUacledProblemTemplate(), DistilacledProblemTemplate()]
        #     return results
        # for timeseries forcating and semi problem, not use MeanBaseline template, it will make meanbaseline to be top rank

        # update v2020.1.28: try sk dummy
        not_add_mean_base_line_task_types = [TaskKeyword.TIME_SERIES.name, TaskKeyword.SEMISUPERVISED.name]
        if not have_intersection(task, not_add_mean_base_line_task_types):
            results.append(SKDummyTemplate())  # put the meanbaseline here so whatever dataset will have a result
        _logger.info("Will add SK Dummy template.")

        for template_class in self.templates:
            template = template_class()
            # sourceType refer to d3m/container/dataset.py ("SEMANTIC_TYPES" as line 40-70)
            # taskType and taskSubtype refer to d3m/
            if have_intersection(task, template.template['taskType']) and have_intersection(subtype, template.template['taskSubtype']):
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

        # filter based on specialized problem
        if specialized_problem == SpecializedProblem.NONE:
            results = [template for template in results
                       if 'specializedProblem' not in template.template]
        else:
            _logger.debug(f'Specialized problem: {specialized_problem}')
            results = [template for template in results
                       if 'specializedProblem' in template.template
                       and specialized_problem in template.template['specializedProblem']]

        # if we finally did not find a proper template to use
        if results == []:
            _logger.error(f"Cannot find a suitable template type to fit the problem: {task}")
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
        """
            loads all templates from template/template_files/loaded
        """
        for each_module_name in sys.modules.keys():
            if each_module_name.startswith("dsbox.template.template_files.loaded."):
                class_name = each_module_name.replace("dsbox.template.template_files.loaded.", "")
                each_module = sys.modules[each_module_name]
                each_template_class = getattr(each_module, class_name)
                self.templates.append(each_template_class)
                self.all_templates[class_name]: each_template_class
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
