import os
import json 
import pickle
import typing

from networkx import nx

from d3m.container.dataset import Dataset
from d3m.metadata.pipeline import Pipeline

from dsbox.template.runtime import Runtime
from dsbox.template.search import ConfigurationSpace, ConfigurationPoint
from dsbox.template.template import to_digraph
import pprint

# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)
TP = typing.TypeVar('TP', bound='FittedPipeline')

class FittedPipeline:
    """
    Fitted pipeline
    Attributes
    ----------
    pipeline: Pipeline
        a pipeline
    dataset: Dataset
        identifier for a dataset
    runtime: Runtime
        runtime containing fitted primitives
    id: str
        the id of the pipeline
    folder_loc: str
        the location of the files of pipeline
    """

    def __init__(self, pipeline = None, runtime = None, dataset = None):
        self.dataset = dataset
        self.runtime = runtime
        self.pipeline = pipeline
        self.id = self.pipeline.id
        self.folder_loc = ''

    @classmethod
    def create(cls:typing.Type[TP], configuration: ConfigurationPoint[PythonPath], dataset: Dataset) -> TP:
        '''
        Initialize the FittedPipeline with the configurations
        '''
        pipeline_to_load = configuration.data['pipeline']
        run = configuration.data['runtime']
        fitted_pipeline_loaded = cls(pipeline_to_load, run, dataset)
        return fitted_pipeline_loaded


    def save(self, folder_loc : str) -> None:
        '''
        Save the given fitted pipeline from TemplateDimensionalSearch
        '''
        self.folder_loc = folder_loc
        # print("The pipeline files will be stored in:")
        # print(self.folder_loc)

        pipeline_dir = os.path.join(self.folder_loc, 'pipelines')
        executable_dir = os.path.join(self.folder_loc, 'executables')
        os.makedirs(pipeline_dir, exist_ok=True)
        os.makedirs(executable_dir, exist_ok=True)

        # print("Writing:",self)


        # save the pipeline with json format
        json_loc = os.path.join(pipeline_dir, self.id + '.json')
        with open(json_loc, 'w') as f:
            self.pipeline.to_json_content(f)

        # save the pickle files of each primitive step
        for i in range(0, len(self.runtime.execution_order)):
            # print("Now saving step_", i)
            n_step = self.runtime.execution_order[i]
            each_step = self.runtime.pipeline[n_step]
            '''
            NOTICE:
            running both of get_params and hyperparams will cause the error of 
            "AttributeError: 'RandomForestClassifier' object has no attribute 'oob_score_'"
            print(each_primitive.get_params())
            print(each_step.hyperparams)
            '''
            file_loc = os.path.join(executable_dir, self.id + "_step_" + str(i) + ".pkl")
            with open(file_loc, "wb") as f:
                pickle.dump(each_step, f)

    def __str__(self):
        desc = list(map(lambda s: (s.primitive, s.hyperparams),
                        self.pipeline.steps))
        return pprint.pformat(desc)
        # print("Sorted:", dag_order)
        # return str(dag_order)

    @classmethod
    def load(cls:typing.Type[TP], folder_loc: str, pipeline_id: str, dataset: Dataset) -> TP:
        '''
        Load the pipeline with given pipeline id and folder location
        '''
        # load pipeline from json
        pipeline_dir = os.path.join(self.folder_loc, 'pipelines')
        executable_dir = os.path.join(self.folder_loc, 'executables')

        json_loc = os.path.join(pipeline_dir, pipeline_id + '.json')
        print("The following pipeline files will be loaded:")
        print(json_loc)
        with open(json_loc, 'r') as f:
            pipeline_to_load = Pipeline.from_json_content(f)

        # load detail fitted parameters from pkl files
        run = Runtime(pipeline_to_load)

        for i in range(0, len(run.execution_order)):
            print("Now loading step", i)
            n_step = run.execution_order[i]
            file_loc = os.path.join(executable_dir, pipeline_id + "_step_" + str(i) + ".pkl")
            with open(file_loc, "rb") as f:
                each_step = pickle.load(f)
                run.pipeline[n_step] = each_step

        fitted_pipeline_loaded = cls(pipeline_to_load, run, dataset)
        return fitted_pipeline_loaded

