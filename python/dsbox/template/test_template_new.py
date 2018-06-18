import json
import os
import glob
import sys
import typing
# os.sys.path.append("/nfs1/dsbox-repo/muxin/dsbox-ta2/python")
# os.sys.path.insert(0, "/nfs1/dsbox-repo/muxin/dsbox-ta2/python")
os.sys.path.insert(0, "/opt/kyao-repo/dsbox2/dsbox-ta2/python")
from enum import Enum

from d3m import utils, index
from d3m.container.dataset import D3MDatasetLoader, Dataset
from d3m.metadata.pipeline import PrimitiveStep, ArgumentType
from d3m.metadata.problem import parse_problem_description, TaskType, TaskSubtype
from d3m.metadata import base as metadata_base

from importlib import reload
import dsbox.template.template
import dsbox.template.library
import dsbox.template.search
reload(dsbox.template.template)
reload(dsbox.template.library)
reload(dsbox.template.search)
from dsbox.template.template import TemplatePipeline, TemplateStep, DSBoxTemplate
from dsbox.template.library import DefaultClassificationTemplate, DefaultRegressionTemplate
from dsbox.template.configuration_space import ConfigurationSpace, SimpleConfigurationSpace
import dsbox.template.search as search

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
pipeline = t.to_pipeline(point)

path = '/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data/seed_datasets_current/38_sick/38_sick_dataset/datasetDoc.json'
path = 'file://{path_schema}'.format(path_schema=os.path.abspath(path))
dataset = D3MDatasetLoader()
dataset = dataset.load(dataset_uri=path)

semantic_types = ["https://metadata.datadrivendiscovery.org/types/CategoricalData",
                  "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                  "https://metadata.datadrivendiscovery.org/types/Target",
                  "https://metadata.datadrivendiscovery.org/types/TrueTarget"]
dataset.metadata = dataset.metadata.update(('0', metadata_base.ALL_ELEMENTS, 30), {'semantic_types': semantic_types})

problem = parse_problem_description('/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data/seed_datasets_current/38_sick/38_sick_problem/problemDoc.json')
metrics = problem['problem']['performance_metrics']

s = search.TemplateDimensionalSearch(t, c, index.search(), dataset, dataset, metrics)
point, value = s.search_one_iter()

print('====classification', point, value)

rpath = '/nfs1/dsbox-repo/data/datasets/seed_datasets_current/196_autoMpg/196_autoMpg_dataset/datasetDoc.json'
rpath = 'file://{path_schema}'.format(path_schema=os.path.abspath(rpath))
rdataset = D3MDatasetLoader()
rdataset = rdataset.load(dataset_uri=rpath)


semantic_types = ["http://schema.org/Float",
                  "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                  "https://metadata.datadrivendiscovery.org/types/Target",
                  "https://metadata.datadrivendiscovery.org/types/TrueTarget"]
rdataset.metadata = rdataset.metadata.update(('0', metadata_base.ALL_ELEMENTS, 8), {'semantic_types': semantic_types})

rproblem = parse_problem_description('/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data/seed_datasets_current/196_autoMpg/196_autoMpg_problem/problemDoc.json')
rmetrics = rproblem['problem']['performance_metrics']

r = DefaultRegressionTemplate()
rc = r.generate_configuration_space()
rpoint = rc.get_point_using_first_value()
rpipeline = r.to_pipeline(rpoint)
rs = search.TemplateDimensionalSearch(r, rc, index.search(), rdataset, rdataset, rmetrics)
rpoint, rvalue = rs.search_one_iter()

print('====regression', rpoint, rvalue)

# print(c._dimension_ordering, cc._dimension_ordering)
# print(c._dimension_values, cc._dimension_values)
