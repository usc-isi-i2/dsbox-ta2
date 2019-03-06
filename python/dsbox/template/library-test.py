"""
Test template library
"""
# import os
# os.sys.path.append('/nas/home/kyao/kyao-repo/dsbox2/dsbox-ta2/python')

import d3m
from d3m.metadata import base as metadata_base
from d3m.metadata.problem import parse_problem_description, TaskType, TaskSubtype
from d3m.container.dataset import D3MDatasetLoader, Dataset

from importlib import reload

import dsbox.template.template as template
reload(template)

import dsbox.template.library as library
reload(library)

import dsbox.template.search as search
reload(search)

template_library = library.TemplateLibrary()
classifer_template_descrptions = template_library.get_templates(TaskType.CLASSIFICATION, None)
template_descrption = classifer_template_descrptions[0]
print(template_descrption.template.template_nodes.keys())

space = search.SimpleConfigurationSpace({'modeller' : ['d3m.primitives.classification.random_forest.SKlearn', 'd3m.primitives.classification.sgd.SKlearn']})

path = '/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data/seed_datasets_current/38_sick/TRAIN/dataset_TRAIN/datasetDoc.json'
path = 'file://{path_schema}'.format(path_schema=os.path.abspath(path))
dataset = D3MDatasetLoader()
dataset = dataset.load(dataset_uri=path)

semantic_types = ["https://metadata.datadrivendiscovery.org/types/CategoricalData",
                  "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                  "https://metadata.datadrivendiscovery.org/types/Target",
                  "https://metadata.datadrivendiscovery.org/types/TrueTarget"]
dataset.metadata = dataset.metadata.update(('0', metadata_base.ALL_ELEMENTS, 30), {'semantic_types': semantic_types})

problem = parse_problem_description('/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data/seed_datasets_current/38_sick/TRAIN/problem_TRAIN/problemDoc.json')
metrics = problem['problem']['performance_metrics']

# Run dimensional search
td_search = search.TemplateDimensionalSearch(template_descrption, space, d3m.index.search(), dataset, dataset, metrics)
point, value = td_search.search_one_iter()
print(point, value)
print(point.data)

# Run dimensional search with random hyperparam generation
rh_search = search.TemplateDimensionalRandomHyperparameterSearch(template_descrption, space, d3m.index.search(), dataset, dataset, metrics)
point, value = rh_search.search_one_iter()

