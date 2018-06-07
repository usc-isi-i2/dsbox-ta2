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
from dsbox.template.search import ConfigurationSpace, ConfigurationPoint
from .template import TemplatePipeline
from .library import TemplateDescription

# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)


class PipelineSave:
    def __init__(self, configuration: ConfigurationPoint[PythonPath], folder_loc: str) -> None:
        self.configuration = configuration
        self.data = configuration.data
        self.runtime = configuration.data['runtime']
        self.pipeline = configuration.data['pipeline']
        self.id = self.pipeline.id
        self.folder_loc = folder_loc


    def save(self) -> None:
        print("The pipeline files will be stored in:")
        print(self.folder_loc)

        # save the pipeline with json format
        json_loc = self.folder_loc + '/pipelines/' + self.id + '.json'
        with open(json_loc, 'w') as f:
            self.pipeline.to_json_content(f)

        pkl_loc = self.folder_loc + '/excutables/' + self.id
        for i in range(0, len(self.runtime.execution_order)):
            print("Now saving step_", i)
            n_step = self.runtime.execution_order[i]
            each_step = self.runtime.pipeline[n_step]
            '''
            NOTICE:
            runing both of get_params and hyperparams will cause the error of 
            "AttributeError: 'RandomForestClassifier' object has no attribute 'oob_score_'"
            print(each_primitive.get_params())
            print(each_step.hyperparams)
            '''
            file_loc = pkl_loc + "_step_" + str(i) + ".pkl"
            # output the pickle file of each step
            with open(file_loc, "wb") as f:
                pickle.dump(each_step, f)

class PipelineLoad:
    def __init__(self, folder_loc: str, pipeline_id: str) -> None:
        self.folder_loc = folder_loc
        self.pipeline_id = pipeline_id

    def load(self) -> Runtime:
        print("The following pipeline files will be loaded:")
        pipeline_loc = self.folder_loc + '/' + self.pipeline_id + '.json'
        print(pipeline_loc)
