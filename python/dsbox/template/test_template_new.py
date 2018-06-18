import json
import os
import glob
import sys
import typing
# os.sys.path.append("/nfs1/dsbox-repo/muxin/dsbox-ta2/python")
os.sys.path.insert(0, "/nfs1/dsbox-repo/muxin/dsbox-ta2/python")
from dsbox.template.library import DefaultClassificationTemplate, DefaultRegressionTemplate
from enum import Enum

from d3m import utils, index
from d3m.container.dataset import Dataset
from d3m.metadata.pipeline import PrimitiveStep, ArgumentType
from d3m.metadata.problem import TaskType, TaskSubtype
from dsbox.template.template import TemplatePipeline, TemplateStep, DSBoxTemplate
from dsbox.template.configuration_space import ConfigurationSpace, SimpleConfigurationSpace
# import dsbox.template.search as search
# import dsbox.template.template as template


def printpipeline(pipeline):
    print("id", pipeline.id)
    print("name", pipeline.name)
    print("context", pipeline.context)
    print("steps: ")
    for s in pipeline.steps:
        print(s.primitive)
    for s in pipeline.outputs:
        print(s)
    for s in pipeline.inputs:
        print(s)


classifier_conf = {
    "denormalize_step": {
        "primitive": "d3m.primitives.datasets.Denormalize",
        "hyperparameters": {},
    },
    "to_dataframe_step": {
        "primitive": "d3m.primitives.datasets.DatasetToDataFrame",
        "hyperparameters": {}
    },
    "column_parser_step": {
        "primitive": "d3m.primitives.data.ColumnParser",
        "hyperparameters": {}
    },
    "extract_attribute_step": {
        "primitive": "d3m.primitives.data.ExtractAttributes",
        "hyperparameters": {}
    },
    "cast_1_step": {
        "primitive": "d3m.primitives.data.CastToType",
        "hyperparameters": {}
    },
    "impute_step": {
        "primitive": "d3m.primitives.sklearn_wrap.SKImputer",
        "hyperparameters": {}
    },
    "extract_target_step": {
        "primitive": "d3m.primitives.data.ExtractTargets",
        "hyperparameters": {}
    },
    "cast_2_step": {
        "primitive": "d3m.primitives.data.CastToType",
        "hyperparameters": {}
    },
    "model_step": {
        "primitive": "d3m.primitives.common_primitives.RandomForestClassifier",
        "hyperparameters": {"n_estimators": {"type": "value", "value": 15}}}
}

regressor_conf = {
    "denormalize_step": {
        "primitive": "d3m.primitives.datasets.Denormalize",
        "hyperparameters": {},
    },
    "to_dataframe_step": {
        "primitive": "d3m.primitives.datasets.DatasetToDataFrame",
        "hyperparameters": {}
    },
    "column_parser_step": {
        "primitive": "d3m.primitives.data.ColumnParser",
        "hyperparameters": {}
    },
    "extract_attribute_step": {
        "primitive": "d3m.primitives.data.ExtractAttributes",
        "hyperparameters": {}
    },
    "cast_1_step": {
        "primitive": "d3m.primitives.data.CastToType",
        "hyperparameters": {}
    },
    "impute_step": {
        "primitive": "d3m.primitives.sklearn_wrap.SKImputer",
        "hyperparameters": {}
    },
    "extract_target_step": {
        "primitive": "d3m.primitives.data.ExtractTargets",
        "hyperparameters": {}
    },
    "model_step": {
        "primitive": "d3m.primitives.sklearn_wrap.SKSGDRegressor",
        "hyperparameters": {}
    }
}


def defaultclassifieryaml():
    t = DefaultClassificationTemplate()
    p = t.to_pipeline(classifier_conf)
    printpipeline(p)
#
    p.check()  # check if the pipeline is valid
# # print(p._context_to_json())
    with open("classfier.yaml", "w") as y:
        p.to_yaml_content(y)


def defaultregressoryaml():
    t = DefaultRegressionTemplate()
    p = t.to_pipeline(regressor_conf)
    printpipeline(p)
#
    p.check()  # check if the pipeline is valid
# # print(p._context_to_json())
    with open("regressor.yaml", "w") as y:
        p.to_yaml_content(y)


# defaultregressoryaml()
t = DefaultClassificationTemplate()
c = t.generate_configuration_space()
point = c.get_point_using_first_value()
print(point.space)
print(point.data)
# pipeline = t.to_pipeline(point)

l = DefaultRegressionTemplate()
cc = l.generate_configuration_space()
print(c._dimension_ordering, cc._dimension_ordering)
print(c._dimension_values, cc._dimension_values)
