import os
import json
import pickle
import pprint
import sys
import typing
import uuid

from d3m.metadata.pipeline import Pipeline
from d3m.metadata.pipeline import StepBase
from d3m.metadata.problem import PerformanceMetric

from dsbox.template.runtime import Runtime

from .utils import larger_is_better

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
        runtime object for the pipeline
    id: str
        the id of the pipeline
    folder_loc: str
        the location of the files of pipeline
    """

    def __init__(self, pipeline: Pipeline, dataset_id: str, log_dir: str, *, id: str = None, metric_descriptions: typing.List = []) -> None:

        # these two are mandatory
        # TODO add the check
        self.dataset_id: str = dataset_id
        self.pipeline: Pipeline = pipeline

        if id is None:
            # Create id distinct, since there may be several fitted pipelines
            #  using the same pipeline
            self.id = str(uuid.uuid4())
        else:
            self.id = id

        self.log_dir = log_dir

        self.runtime = Runtime(pipeline, self.id, self.log_dir)

        self.metric_descriptions = list(metric_descriptions)
        self.runtime.set_metric_descriptions(self.metric_descriptions)

        self.metric: typing.Dict = {}

        self.auxiliary: typing.Dict = {}

    def _set_fitted(self, fitted_pipe: typing.List[StepBase]) -> None:
        self.runtime.pipeline = fitted_pipe

    # @classmethod
    # def create(cls: typing.Type[TP],
    #            configuration:ConfigurationPoint,
    #            dataset: Dataset) -> TP:
    #     '''
    #     Initialize the FittedPipeline with the configurations
    #     '''
    #
    #     assert False, "This method is deprecated!"
    #
    #     # pipeline_to_load = template.to_pipeline(configuration)
    #     # run = []#configuration.data['runtime']
    #     fitted_pipe = configuration.data['fitted_pipe']
    #     pipeline = configuration.data['pipeline']
    #     exec_order = configuration.data['exec_plan']
    #
    #
    #     fitted_pipeline_loaded = cls(
    #         fitted_pipe=fitted_pipe,
    #         pipeline=pipeline,
    #         exec_order=exec_order,
    #         dataset_id=dataset.metadata.query(())['id']
    #     )
    #     return fitted_pipeline_loaded

    def set_metric(self, metric: typing.Dict):
        self.metric = metric

    def fit(self, **arguments):
        self.runtime.fit(**arguments)

    def produce(self, **arguments):
        self.runtime.produce(**arguments)

    def get_cross_validation_metrics(self) -> typing.List:
        return self.runtime.cross_validation_result

    def get_fit_step_output(self, step_number: int):
        return self.runtime.fit_outputs[step_number]

    def get_produce_step_output(self, step_number: int):
        return self.runtime.produce_outputs[step_number]

    def save(self, folder_loc : str) -> None:
        '''
        Save the given fitted pipeline from TemplateDimensionalSearch
        '''
        pipeline_dir = os.path.join(folder_loc, 'pipelines')
        executable_dir = os.path.join(folder_loc, 'executables')
        supporting_files_dir = os.path.join(folder_loc, 'supporting_files',
                                            self.id)
        os.makedirs(pipeline_dir, exist_ok=True)
        os.makedirs(executable_dir, exist_ok=True)
        os.makedirs(supporting_files_dir, exist_ok=True)

        # print("Writing:",self)

        # store fitted_pipeline id
        structure = self.pipeline.to_json_structure()
        structure['fitted_pipeline_id'] = self.id
        structure['dataset_id'] = self.dataset_id

        # Save pipeline rank
        if self.metric:
            metric: str = self.metric['metric']
            value: float = self.metric['value']
            if larger_is_better(metric):
                if value == 0.0:
                    rank = sys.float_info.max
                else:
                    rank = 1/value
            else:
                rank = value
        structure['pipeline_rank'] = rank
        structure['metric'] = metric
        structure['metric_value'] = value

        # FIXME: this is here for testing purposes
        # structure['runtime_stats'] = str(self.auxiliary)

        # save the pipeline with json format
        json_loc = os.path.join(pipeline_dir, self.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out)

        # save the pipeline spec under executables to be a json file simply specifies the pipeline id.
        json_loc = os.path.join(executable_dir, self.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump({"fitted_pipeline_id": self.id}, out)

        # save the pickle files of each primitive step
        for i in range(0, len(self.runtime.execution_order)):
            # print("Now saving step_", i)
            # n_step = self.runtime.execution_order[i]
            each_step = self.runtime.pipeline[i]
            file_loc = os.path.join(supporting_files_dir,
                                    "step_" + str(i) + ".pkl")
            with open(file_loc, "wb") as f:
                pickle.dump(each_step, f)

    def __str__(self):
        # desc = list(map(lambda s: (s.primitive, s.hyperparams),
        #                 ))
        return pprint.pformat(self.runtime.pipeline)
        # print("Sorted:", dag_order)
        # return str(dag_order)

    @classmethod
    def load(cls:typing.Type[TP], folder_loc: str,
             pipeline_id: str, log_dir: str, dataset_id: str = None,) -> typing.Tuple[TP, Runtime]:
        '''
        Load the pipeline with given pipeline id and folder location
        '''
        # load pipeline from json
        pipeline_dir = os.path.join(folder_loc, 'pipelines')
        executable_dir = os.path.join(folder_loc, 'executables')

        pipeline_spec_loc = os.path.join(executable_dir, pipeline_id + '.json')

        with open(pipeline_spec_loc, 'r') as f:
            fitted_pipeline_id = json.load(f).get('fitted_pipeline_id')

        pipeline_definition_loc = os.path.join(pipeline_dir,
                                               fitted_pipeline_id + '.json')
        print("The following pipeline file will be loaded:")
        print(pipeline_definition_loc)

        with open(pipeline_definition_loc, 'r') as f:
            structure = json.load(f)

        dataset_id = structure.get('dataset_id')

        pipeline_to_load = Pipeline.from_json_structure(structure)

        # load detail fitted parameters from pkl files in
        # supporting_files/<fitted_pipeline_id>
        supporting_files_dir = os.path.join(folder_loc, 'supporting_files',
                                            fitted_pipeline_id)

        run = Runtime(pipeline_to_load, fitted_pipeline_id, log_dir)

        for i in range(0, len(run.execution_order)):
            # print("Now loading step", i)
            file_loc = os.path.join(supporting_files_dir,
                                    "step_" + str(i) + ".pkl")
            with open(file_loc, "rb") as f:
                each_step = pickle.load(f)
                run.pipeline[i] = each_step


        # fitted_pipeline_loaded = cls(pipeline_to_load, run, dataset)
        fitted_pipeline_loaded = cls(pipeline=pipeline_to_load,
                                     dataset_id=dataset_id,
                                     id=fitted_pipeline_id,
                                     log_dir=log_dir)
        fitted_pipeline_loaded._set_fitted(run.pipeline)

        return (fitted_pipeline_loaded, run)

    def __getstate__(self) -> typing.Dict:
        """
        This method is used by the pickler as the state of object.
        The object can be recovered through this state uniquely.

        Returns:
            state: Dict
                dictionary of important attributes of the object

        """
        # print("[INFO] Get state called")

        state = self.__dict__  # get attribute dictionary

        # add the fitted_primitives
        state['fitted_pipe'] = self.runtime.pipeline
        state['pipeline'] = self.pipeline.to_json_structure()
        state['log_dir'] = self.log_dir
        state['id'] = self.id
        del state['runtime']  # remove runtime entry

        return state

    def __setstate__(self, state: typing.Dict) -> None:
        """
        This method is used for unpickling the object. It takes a dictionary
        of saved state of object and restores the object to that state.
        Args:
            state: typing.Dict
                dictionary of the objects picklable state
        Returns:

        """

        # print("[INFO] Set state called!")

        fitted = state['fitted_pipe']
        del state['fitted_pipe']

        structure = state['pipeline']
        state['pipeline'] = Pipeline.from_json_structure(structure)

        run = Runtime(state['pipeline'], state['id'], state['log_dir'])
        run.pipeline = fitted

        state['runtime'] = run

        self.__dict__ = state
