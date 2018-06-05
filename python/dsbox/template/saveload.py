import abc
import bisect
import operator
import random
import typing

import d3m.exceptions as exceptions
import pickle
import dill
from d3m import utils
from d3m.container.dataset import Dataset
from d3m.metadata.base import PrimitiveMetadata
from d3m.primitive_interfaces.base import PrimitiveBaseMeta
from d3m.runtime import Runtime
from d3m.metadata.pipeline import Pipeline
from dsbox.schema.problem import optimization_type, OptimizationType
from dsbox.template.search import ConfigurationSpace
from .template import TemplatePipeline
from .library import TemplateDescription


class PipelineSave:
    def __init__(self, configuration_space: ConfigurationSpace) -> None:

        self.configuration_space = configuration_space

    def save_to(self, folder_loc: str) -> None:
        print("The pipeline files will be stored in:")
        print(folder_loc)


class PipelineLoad:
    def __init__(self, configuration_space: ConfigurationSpace) -> None:

        self.configuration_space = configuration_space

    def save_to(self, folder_loc: str) -> None:
        print("The pipeline files will be stored in:")
        print(folder_loc)
