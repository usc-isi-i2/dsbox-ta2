import json
import os
import pickle
import pprint
import typing
import uuid

from d3m.metadata.pipeline import Pipeline

from dsbox.template.runtime import Runtime
from dsbox.template.configuration_space import ConfigurationPoint

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
    dataset_id: str
        identifier for a dataset (to get dataset id, use dataset.metadata.query(())['id'])
    runtime: Runtime
        runtime containing fitted primitives
    id: str
        the id of the pipeline
    """

    def __init__(self, pipeline:Pipeline, *, runtime:Runtime = None,
                 dataset_id:str = None, id:str = None) -> None:
        self.pipeline = pipeline
        self.dataset_id = dataset_id

        if runtime is None:
            self.runtime = Runtime(pipeline)
        else:
            self.runtime = runtime

        if id is None:
            # Create id distinct, since there may be several fitted pipelines using the same pipeline
            self.id = str(uuid.uuid4())
        else:
            self.id = id

    def fit(self, **arguments):
        self.runtime.fit(**arguments)

    def produce(self, **arguments):
        self.runtime.produce(**arguments)

    def get_fit_step_output(self, step_number: int):
        return self.runtime.fit_outputs[step_number]

    def get_produce_step_output(self, step_number: int):
        return self.runtime.produce_outputs[step_number]

    @classmethod
    def create(cls:typing.Type[TP], configuration: ConfigurationPoint[PythonPath], dataset_id: str) -> TP:
        '''
        Initialize the FittedPipeline with the configurations
        '''
        pipeline_to_load = configuration.data['pipeline']
        run = configuration.data['runtime']
        fitted_pipeline_loaded = cls(pipeline_to_load, runtime=run, dataset_id=dataset_id)
        return fitted_pipeline_loaded


    def save(self, folder_loc : str) -> None:
        '''
        Save the given fitted pipeline from TemplateDimensionalSearch
        '''
        # print("The pipeline files will be stored in:")
        # print(folder_loc)

        pipeline_dir = os.path.join(folder_loc, 'pipelines')
        executable_dir = os.path.join(folder_loc, 'executables', self.id)
        os.makedirs(pipeline_dir, exist_ok=True)
        os.makedirs(executable_dir, exist_ok=True)

        # print("Writing:",self)

        json_loc = os.path.join(pipeline_dir, self.id + '-1.json')
        with open(json_loc, 'w') as out:
            self.pipeline.to_json(out)

        # store fitted_pipeline id
        structure = self.pipeline.to_json_structure()
        structure['fitted_pipeline_id'] = self.id
        structure['dataset_id'] = self.dataset_id

        # save the pipeline with json format
        json_loc = os.path.join(pipeline_dir, self.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out)

        # save the pickle files of each primitive step
        for i in range(0, len(self.runtime.execution_order)):
            print("Now saving step_", i)
            #n_step = self.runtime.execution_order[i]
            each_step = self.runtime.pipeline[i]
            file_loc = os.path.join(executable_dir, "step_" + str(i) + ".pkl")
            with open(file_loc, "wb") as f:
                pickle.dump(each_step, f)

    def __str__(self):
        desc = list(map(lambda s: (s.primitive, s.hyperparams),
                        self.pipeline.steps))
        return pprint.pformat(desc)
        # print("Sorted:", dag_order)
        # return str(dag_order)

    @classmethod
    def load(cls:typing.Type[TP], folder_loc: str, pipeline_id: str, dataset_id: str = None) -> TP:
        '''
        Load the pipeline with given pipeline id and folder location
        '''
        # load pipeline from json
        pipeline_dir = os.path.join(folder_loc, 'pipelines')
        executable_dir = os.path.join(folder_loc, 'executables', pipeline_id)

        json_loc = os.path.join(pipeline_dir, pipeline_id + '.json')
        print("The following pipeline file will be loaded:")
        print(json_loc)
        with open(json_loc, 'r') as f:
            structure = json.load(f)

        fitted_pipeline_id = structure['fitted_pipeline_id']
        dataset_id = structure['dataset_id']

        pipeline_to_load = Pipeline.from_json_structure(structure)

        # load detail fitted parameters from pkl files
        run = Runtime(pipeline_to_load)

        for i in range(0, len(run.execution_order)):
            print("Now loading step", i)
            file_loc = os.path.join(executable_dir, "step_" + str(i) + ".pkl")
            with open(file_loc, "rb") as f:
                each_step = pickle.load(f)
                run.pipeline[i] = each_step

        fitted_pipeline_loaded = cls(pipeline=pipeline_to_load, runtime=run,
                                     dataset_id=dataset_id, id=fitted_pipeline_id)
        return fitted_pipeline_loaded
