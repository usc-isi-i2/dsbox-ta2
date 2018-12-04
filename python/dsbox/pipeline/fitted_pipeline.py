import os
import json
import logging
import pickle
import pprint
import sys
import typing
import uuid

from d3m.metadata.pipeline import Pipeline, Resolver, StepBase, PrimitiveStep, SubpipelineStep
from d3m.metadata.problem import PerformanceMetric
from d3m import exceptions
from dsbox.template.runtime import Runtime, ForkedPdb

from .utils import larger_is_better

TP = typing.TypeVar('TP', bound='FittedPipeline')

_logger = logging.getLogger(__name__)


class FittedPipeline:
    """
    Fitted pipeline
    Attributes
    ----------
    dataset_id: str
        The string type mark to define the dataset of this pipeline used
    pipeline: Pipeline
        a pipeline
    template/problem:
        Used to add extra explanation, not necessary to be given
    runtime: Runtime
        runtime object for the pipeline
    log_dir: str
        the direction to store the log files
    id: str
        the id of the pipeline
    metric_descriptions: List[str]
        describe the type of the metrics, needed for calculating the score of the predictions
    auxiliary: dict
        detail processes of fitted pipeline, not necessaary to be added
    """

    def __init__(self, pipeline: Pipeline, dataset_id: str, log_dir: str, *, id: str = None,
                 metric_descriptions: typing.List = [], template=None, problem=None) -> None:

        # these two are mandatory
        # TODO add the check
        self.dataset_id: str = dataset_id
        self.pipeline: Pipeline = pipeline
        self.template = template
        self.problem = problem
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

        _logger.debug('Creating fitted pipeline %s', self.id)

    def _set_fitted(self, fitted_pipe: typing.List[StepBase]) -> None:
        self.runtime.steps_state = fitted_pipe

    def set_metric(self, metric: typing.Dict):
        """
            Set the metric type for this fitted pipeline
        """
        self.metric = metric

    def fit(self, **arguments):
        _logger.debug('Fitting fitted pipeline %s', self.id)
        self.runtime.fit(**arguments)

    def produce(self, **arguments):
        _logger.debug('Producing fitted pipeline %s', self.id)
        self.runtime.produce(**arguments)

    def get_cross_validation_metrics(self) -> typing.List:
        return self.runtime.cross_validation_result

    def get_fit_step_output(self, step_number: int = 0):
        #return self.runtime.fit_outputs[step_number]
        # Fix it: should here always be 0?
        return self.runtime.fit_outputs[0] 

    def get_produce_step_output(self, step_number: int):
        # return self.runtime.produce_outputs[step_number]
        return self.runtime.produce_outputs[0] 

    def save(self, folder_loc: str) -> None:
        '''
        Save the given fitted pipeline from TemplateDimensionalSearch
        inputs:
        folder_loc: str
            the location of the files of pipeline
        '''
        _logger.debug('Saving fitted pipeline %s', self.id)
        pipeline_dir = os.path.join(folder_loc, 'pipelines')
        executable_dir = os.path.join(folder_loc, 'executables')
        supporting_files_dir = os.path.join(folder_loc, 'supporting_files')
        os.makedirs(pipeline_dir, exist_ok=True)
        os.makedirs(executable_dir, exist_ok=True)
        os.makedirs(supporting_files_dir, exist_ok=True)

        # store fitted_pipeline id
        structure = self.pipeline.to_json_structure()
        structure["parent_id"] = self.pipeline.id
        structure['id'] = self.id
        structure['dataset_id'] = self.dataset_id
        # add timing for each step
        # for each_step in structure['steps']:
        #     primitive_name = each_step["primitive"]["python_path"]
        #     each_step["timing"] = self.runtime.timing[primitive_name]
        # FIXME [TIMING]
        # add timing for each step
        # for each_step in structure['steps']:
        #     primitive_name = each_step["primitive"]["python_path"]
        #     each_step["timing"] = self.runtime.timing[primitive_name]

        # Save pipeline rank
        if self.metric:
            metric: str = self.metric['metric']
            value: float = self.metric['value']
            if larger_is_better(metric):
                if value == 0.0:
                    rank = sys.float_info.max
                else:
                    rank = 1 / value
            else:
                rank = value
            structure['pipeline_rank'] = rank
            structure['metric'] = metric
            structure['metric_value'] = value
        else:
            _logger.warn("[WARN] Metric type of the pipeline is unknown, unable to calculate the rank of the pipeline")

        if self.template:
            structure['template_name'] = self.template.template['name']
            structure['template_taskType'] = str(self.template.template['taskType'])
            structure['template_taskSubtype'] = str(self.template.template['taskSubtype'])
        else:
            _logger.warn("[WARN] Template type of the pipeline is unknown, unable to save template name / taskType / taskSubtype")

        if self.problem:
            problem_meta = self.problem.query(())['about']
            structure['problem_taskType'] = str(problem_meta['taskType'])
            try:
                structure['problem_taskSubType'] = str(problem_meta['taskSubType'])
            except:
                structure['problem_taskSubType'] = "NONE"
        else:
            _logger.warn("[WARN] problem type of the pipeline is unknown, unable to save problem taskType / taskSubtype")

        # structure['total_time_used'] = self.runtime.timing["total_time_used"]

        # structure['total_time_used_without_cache'] = self.runtime.timing["total_time_used_without_cache"]

        # structure['template'] = runtime.

        # FIXME: this is here for testing purposes
        # structure['runtime_stats'] = str(self.auxiliary)

        # save the pipeline with json format
        json_loc = os.path.join(pipeline_dir, self.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out)

        # save the pipeline spec under executables to be a json file simply specifies the pipeline id
        json_loc = os.path.join(executable_dir, self.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump({"fitted_pipeline_id": self.id}, out)

        # save subpipelines if exists
        for each_step in self.pipeline.steps:
            if isinstance(each_step, SubpipelineStep):
                need_save = True
                json_loc = os.path.join(pipeline_dir, each_step.pipeline.id + '.json')
                subpipeline_structure = each_step.pipeline.to_json_structure()

                # if pipeline already exist, check it
                if os.path.exists(json_loc):
                    with open(json_loc, 'r') as out:
                        temp_pipeline = json.load(out)
                        if 'pipeline_rank' not in temp_pipeline:
                            _logger.warn("The sub-pipeline {} of pipeline {} do not have rank".format(each_step.pipeline.id, self.id))
                        if 'steps' in temp_pipeline:
                            if temp_pipeline['steps'] != subpipeline_structure:
                                _logger.warn("The pipeline structure of {} is not same as new one.".format(each_step.pipeline.id))
                            else:
                                need_save = False
                        else:
                            _logger.warn("The original pipeline file of {} is not completed.".format(each_step.pipeline.id))

                if need_save:
                    with open(json_loc, 'w') as out:
                        json.dump(subpipeline_structure, out)

        # save the detail supporting files
        assert len(self.runtime.steps_state) == len(self.pipeline.steps)
        
        self.save_pickle_files(folder_loc, self.runtime)

    def save_pickle_files(self, folder_loc: str, input_runtime: Runtime) -> None:
        supporting_files_dir = os.path.join(folder_loc, 'supporting_files', input_runtime.fitted_pipeline_id)
        os.makedirs(supporting_files_dir, exist_ok=True)
        # save the pickle files of each primitive step
        for i in range(len(input_runtime.steps_state)):
            # print("Now saving step_", i)
            each_step = input_runtime.steps_state[i]
            if isinstance(each_step, Runtime):
                # if it is a subpipeline recursively call pickling functions
                self.save_pickle_files(folder_loc, each_step)
            else:
                # else if it is a primitive step, pickle directly
                file_loc = os.path.join(supporting_files_dir, "step_" + str(i) + ".pkl")
                with open(file_loc, "wb") as f:
                    pickle.dump(each_step, f)

        _logger.info("Saving pickle files of pipeline {} finished.".format(self.id))

    @classmethod
    def load(cls: typing.Type[TP], folder_loc: str,
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
        _logger.info(f"pipeline file will be loaded: {pipeline_definition_loc}")

        with open(pipeline_definition_loc, 'r') as f:
            structure = json.load(f)

        dataset_id = structure.get('dataset_id')

        resolver = Resolver(pipeline_search_paths = [pipeline_dir])
        pipeline_to_load = Pipeline.from_json_structure(pipeline_description = structure, resolver = resolver)
        structure["id"] = pipeline_to_load.id
        # load detail fitted parameters from pkl files in
        # supporting_files/<fitted_pipeline_id>
        runtime_loaded = FittedPipeline.load_pickle_files(pipeline_to_load, log_dir, folder_loc)

        fitted_pipeline_loaded = cls(pipeline=pipeline_to_load,
                                     dataset_id=dataset_id,
                                     id=fitted_pipeline_id,
                                     log_dir=log_dir)
        fitted_pipeline_loaded._set_fitted(runtime_loaded.steps_state)

        return (fitted_pipeline_loaded, runtime_loaded)

    @classmethod
    def load_pickle_files(cls, pipeline_to_load: Pipeline, log_dir: str, folder_loc: str):
        '''
            create a Runtime instance from given pipelines and supporting files
        '''
        runtime_loaded = Runtime(pipeline_to_load, pipeline_to_load.id, log_dir)
        supporting_files_dir = os.path.join(folder_loc, 'supporting_files', pipeline_to_load.id)
        for i, each_step in enumerate(pipeline_to_load.steps):
            # if it is a primitive, load directly
            if isinstance(each_step, PrimitiveStep):
                file_loc = os.path.join(supporting_files_dir, "step_" + str(i) + ".pkl")
                with open(file_loc, "rb") as f:
                    each_step = pickle.load(f)
                    runtime_loaded.steps_state[i] = each_step

            # if it is a subpipeline, recursively creat a new runtime
            elif isinstance(each_step, SubpipelineStep):
                runtime_loaded.steps_state[i] = FittedPipeline.load_pickle_files(each_step.pipeline, log_dir, folder_loc)

            else:
                raise exceptions.UnexpectedValueError("Unknown types for step " + str(i) + "as" + str(type(each_step)))

        return runtime_loaded

    def __str__(self):
        # desc = list(map(lambda s: (s.primitive, s.hyperparams),
        #                 ))
        return pprint.pformat(self.runtime.steps_state)
        # print("Sorted:", dag_order)
        # return str(dag_order)

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
        state['fitted_pipe'] = self.runtime.steps_state
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
        run.steps_state = fitted

        state['runtime'] = run

        self.__dict__ = state