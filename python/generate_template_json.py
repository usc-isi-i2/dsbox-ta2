import json
import os
import glob
import sys
import typing
os.sys.path.insert(0, "/nfs1/dsbox-repo/muxin/dsbox-ta2/python")

from d3m import utils, index
from d3m.container.dataset import D3MDatasetLoader, Dataset
from d3m.metadata.pipeline import PrimitiveStep, ArgumentType
from d3m.metadata.problem import parse_problem_description, TaskType, TaskSubtype
from d3m.metadata import base as metadata_base


from importlib import reload
import dsbox.template.template
import dsbox.template.library

reload(dsbox.template.template)
reload(dsbox.template.library)

from dsbox.template.template import TemplatePipeline, TemplateStep, DSBoxTemplate
from dsbox.template.library import TemplateLibrary, SRIVertexNominationTemplate, JHUVertexNominationTemplate, SRILinkPredictionTemplate, SRIGraphMatchingTemplate, JHUGraphMatchingTemplate
from dsbox.template.configuration_space import ConfigurationSpace, SimpleConfigurationSpace
import pdb

tmplist = TemplateLibrary()
tmplist.templates = []
# tmplist.templates.append(DefaultGraphMatchingTemplate)
tmplist.templates.append(JHUVertexNominationTemplate)
tmplist.templates.append(JHUGraphMatchingTemplate)
# tmplist.templates.append(DefaultLinkPredictionTemplate)
# tmplist.templates.append(DefaultImageProcessingRegressionTemplate)
for v in tmplist.templates:
    # pdb.set_trace()
    v = v()
    space = v.generate_configuration_space()
    point = space.get_first_assignment()
    # point = c.generate_configuration_space()
# print(point.space)
# print(point.data)
# pdb.set_trace()
    pipeline = v.to_pipeline(point)
    pipeline.check()

    filename = "/nfs1/dsbox-repo/muxin/pipelines/DSBOX/" + v.template["name"] + ".json"
    with open(filename, "w") as f:
        pipeline.to_json(f)
