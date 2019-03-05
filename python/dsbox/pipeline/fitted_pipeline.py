import os
import json
import logging
import pickle
import pprint
import sys
import typing
import uuid

from d3m.metadata.pipeline import Pipeline, Resolver, StepBase, PrimitiveStep, SubpipelineStep
from d3m import exceptions

from dsbox.template.runtime import Runtime,ForkedPdb
from dsbox.template.template import DSBoxTemplate
from dsbox.datapreprocessing.cleaner.splitter import Splitter, SplitterHyperparameter

from .utils import larger_is_better

FP = typing.TypeVar('FP', bound='FittedPipeline')

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

    # Static volume for data need by d3m primitives
    static_volume_dir: str = None

    def __init__(self, pipeline: Pipeline, dataset_id: str, log_dir: str, *, id: str = None,
                 metric_descriptions: typing.List = [], template: DSBoxTemplate = None,
                 template_name: str = None, template_task: str = None, template_subtask: str = None,
                 problem=None) -> None:

        # these two are mandatory
        # TODO add the check
        self.dataset_id: str = dataset_id
        self.pipeline: Pipeline = pipeline
        self.template_name = ""
        self.template_task = ""
        self.template_subtask = ""
        if template is not None:
            self.template_name = template.template['name']
            self.template_task = str(template.template['taskType'])
            self.template_subtask = str(template.template['taskSubtype'])
        if template_name:
            self.template_name = template_name
        if template_task:
            self.template_task = template_task
        if template_subtask:
            self.template_subtask = template_subtask
        self.problem = problem
        if id is None:
            # Create id distinct, since there may be several fitted pipelines
            #  using the same pipeline
            self.id = str(uuid.uuid4())
        else:
            self.id = id

        self.log_dir = log_dir

        self.runtime = Runtime(pipeline, fitted_pipeline_id=self.id, template_name=self.template_name,
                               volumes_dir=FittedPipeline.static_volume_dir, log_dir=self.log_dir)

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

    def set_sampler_primitive(self):
        """
            Set the placeholder-like primitive DoNothingForDataset (at the first step of the pipeline if have)
            to the real primitive (splitter) that should store in the pipeline
        """
        # ForkedPdb().set_trace()
        splitter_metadata = Splitter.metadata.query()
        sampler_primitive_augument= {
                                      "type": "PRIMITIVE",
                                      "primitive": {
                                        "id": splitter_metadata['id'],
                                        "version": splitter_metadata['version'],
                                        "python_path": splitter_metadata['python_path'],
                                        "name": splitter_metadata['name'],
                                        "digest": splitter_metadata['digest']
                                      },
                                      "arguments": {
                                        "inputs": {
                                          "type": "CONTAINER",
                                          "data": "inputs.0"
                                        }
                                      },
                                      "outputs": [
                                        {
                                          "id": "produce"
                                        }
                                      ]
                                    }
        sampler_hyperparams_file_loc = os.path.join(os.environ["D3MLOCALDIR"], "splitter.json")
        with open(sampler_hyperparams_file_loc, "r") as f:
            sampler_hyperparams_file = json.load(f)

        new_hyper_file = {}
        for key, value in sampler_hyperparams_file.items():
            new_hyper_file[key] = {"type":"VALUE",
                                   "data":value}

        sampler_primitive_augument['hyperparams'] =  new_hyper_file

        sampler_step = PrimitiveStep.from_json_structure(step_description = sampler_primitive_augument)
        sampler_pickle_file_loc = os.path.join(os.environ["D3MLOCALDIR"], "splitter.pkl")
        with open(sampler_pickle_file_loc, "rb") as f:
            sampler_pickle_file = pickle.load(f)
        # change pickling file in runtime to be sampler
        self.runtime.steps_state[0] = sampler_pickle_file
        self.pipeline.replace_step(index=0, replacement_step=sampler_step)

    def fit(self, **arguments):
        _logger.debug('Fitting fitted pipeline %s', self.id)
        inputs = arguments['inputs']
        del arguments['inputs']
        self.runtime.fit(inputs, **arguments)

    def produce(self, **arguments):
        _logger.debug('Producing fitted pipeline %s', self.id)
        inputs = arguments['inputs']
        del arguments['inputs']
        self.runtime.produce(inputs, **arguments)

    def get_cross_validation_metrics(self) -> typing.List:
        return self.runtime.cross_validation_result

    def get_fit_step_output(self, step_number: int = 0):
        #return self.runtime.fit_outputs[step_number]
        # TODO: should here always be 0?
        # return self.runtime.fit_outputs[0]
        return self.runtime.fit_outputs.values['outputs.0']

    def get_produce_step_output(self, step_number: int):
        # return self.runtime.produce_outputs[step_number]
        # TODO: should here always be 0?
        # return self.runtime.produce_outputs[0]
        return self.runtime.produce_outputs.values['outputs.0']

    def save_schema_only(self, folder_loc: str, pipeline_schema_subdir: str = 'pipelines_scored',
                         subpipelines_subdir: str = 'subpipelines') -> None:
        '''
        Save the pipline schema and subpline schema only.
        '''
        _logger.debug(f'Save pipeline schema {self.id}')

        pipeline_dir = os.path.join(folder_loc, pipeline_schema_subdir)
        subpipeline_dir = os.path.join(folder_loc, subpipelines_subdir)

        structure = self.pipeline.to_json_structure()
        # if we has DoNothingForDataset, we need update pipeline again
        if structure['steps'][0]['primitive'] ['python_path'] == "d3m.primitives.data_preprocessing.DoNothingForDataset.DSBOX":
            _logger.info("The pipeline with DoNothingForDataset at step.0 detected.")
            self.set_sampler_primitive()
            structure = self.pipeline.to_json_structure()
            _logger.info("The pipeline with DoNothingForDataset has been replaced to be Splitter.")

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
            _logger.warn("Metric type of the pipeline is unknown, unable to calculate the rank of the pipeline")

        if self.runtime.cross_validation_result:
            structure['cross_validation'] = self.runtime.cross_validation_result

        structure['template_name'] = self.template_name
        structure['template_task'] = self.template_task
        structure['template_subtask'] = self.template_subtask

        if self.problem:
            problem_meta = self.problem.query(())['about']
            structure['problem_taskType'] = str(problem_meta['taskType'])
            try:
                structure['problem_taskSubType'] = str(problem_meta['taskSubType'])
            except Exception:
                structure['problem_taskSubType'] = "NONE"
        else:
            _logger.warn("Problem type of the pipeline is unknown, unable to save problem taskType / taskSubtype")

        # save the pipeline with json format
        json_loc = os.path.join(pipeline_dir, self.pipeline.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out)

        # save subpipelines if exists
        for each_step in self.pipeline.steps:
            if isinstance(each_step, SubpipelineStep):
                need_save = True
                json_loc = os.path.join(subpipeline_dir, each_step.pipeline.id + '.json')
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

    def save(self, folder_loc: str, pipeline_schema_subdir: str = 'pipelines_scored',
                             subpipelines_subdir: str = 'subpipelines', pipelines_fitted_subdir: str = 'pipelines_fitted') -> None:
        self.save_schema_only(folder_loc, pipeline_schema_subdir, subpipelines_subdir)

        pipelines_fitted_dir = os.path.join(folder_loc, pipelines_fitted_subdir)
        sub_dir = os.path.join(pipelines_fitted_dir, self.id)
        os.makedirs(sub_dir, exist_ok=True)

        # Save fitted pipeline structure
        structure = {
            'fitted_pipeline_id': self.id,
            'pipeline_id': self.pipeline.id,
            'dataset_id': self.dataset_id
        }
        json_loc = os.path.join(pipelines_fitted_dir, self.id, self.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out)

        # pickle trained primitives
        assert len(self.runtime.steps_state) == len(self.pipeline.steps)
        self.save_pickle_files(pipelines_fitted_dir, self.runtime)

    def save_pickle_files(self, pipelines_fitted_dir: str, input_runtime: Runtime) -> None:
        pipelines_fitted_dir = os.path.join(pipelines_fitted_dir, input_runtime.fitted_pipeline_id)
        os.makedirs(pipelines_fitted_dir, exist_ok=True)
        # save the pickle files of each primitive step
        for i in range(len(input_runtime.steps_state)):
            # print("Now saving step_", i)
            each_step = input_runtime.steps_state[i]
            if isinstance(each_step, Runtime):
                # if it is a subpipeline recursively call pickling functions
                self.save_pickle_files(pipelines_fitted_dir, each_step)
            else:
                # else if it is a primitive step, pickle directly
                file_loc = os.path.join(pipelines_fitted_dir, "step_" + str(i) + ".pkl")
                with open(file_loc, "wb") as f:
                    pickle.dump(each_step, f)

        _logger.info("Saving pickle files of pipeline {} finished.".format(self.id))

    @classmethod
    def load_schema_only(cls: typing.Type[FP], pipeline_id: str, folder_loc: str,
                         pipeline_schema_subdir: str = 'pipelines_scored',
                         subpipelines_subdir: str = 'subpipelines') -> (Pipeline, typing.Dict):
        pipeline_dir = os.path.join(folder_loc, pipeline_schema_subdir)
        subpipeline_dir = os.path.join(folder_loc, subpipelines_subdir)

        pipeline_schema = os.path.join(pipeline_dir, pipeline_id + '.json')

        with open(pipeline_schema, 'r') as f:
            structure = json.load(f)

        resolver = Resolver(pipeline_search_paths=[pipeline_dir, subpipeline_dir])
        pipeline = Pipeline.from_json_structure(pipeline_description=structure, resolver=resolver)
        return (pipeline, structure)

    @classmethod
    def load(cls: typing.Type[FP], *, fitted_pipeline_id: str, folder_loc: str, log_dir: str,
             pipeline_schema_subdir: str = 'pipelines_scored', subpipelines_subdir: str = 'subpipelines',
             pipelines_fitted_subdir: str = 'pipelines_fitted') -> FP:

        pipelines_fitted_dir = os.path.join(folder_loc, pipelines_fitted_subdir)

        fitted_pipeline_schema = os.path.join(pipelines_fitted_dir, fitted_pipeline_id, fitted_pipeline_id+'.json')
        with open(fitted_pipeline_schema, 'r') as f:
            structure = json.load(f)

        pipeline, pipeline_structure = FittedPipeline.load_schema_only(structure['pipeline_id'], folder_loc)
        runtime: Runtime = FittedPipeline.load_pickle_files(
            pipeline, structure['fitted_pipeline_id'], pipelines_fitted_dir=pipelines_fitted_dir, log_dir=log_dir)

        fitted_pipeline = FittedPipeline(pipeline, dataset_id=structure['dataset_id'], log_dir=log_dir, id=fitted_pipeline_id)
        fitted_pipeline.runtime = runtime
        if 'template_name' in pipeline_structure:
            fitted_pipeline.template_name = pipeline_structure['template_name']
            fitted_pipeline.template_task = pipeline_structure['template_task']
            fitted_pipeline.template_subtask = pipeline_structure['template_subtask']
        fitted_pipeline._set_fitted(runtime.steps_state)
        return fitted_pipeline

    @classmethod
    def load_old(cls: typing.Type[FP], folder_loc: str,
                 pipeline_id: str, log_dir: str, dataset_id: str = None,) -> typing.Tuple[FP, Runtime]:

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

        fitted_pipeline_dir = os.path.join(folder_loc, 'supporting_files')
        runtime_loaded = FittedPipeline.load_pickle_files(pipeline_to_load, pipeline_to_load.id, fitted_pipeline_dir, log_dir)

        fitted_pipeline_loaded = cls(pipeline=pipeline_to_load,
                                     dataset_id=dataset_id,
                                     id=fitted_pipeline_id,
                                     log_dir=log_dir)
        fitted_pipeline_loaded._set_fitted(runtime_loaded.steps_state)

        return (fitted_pipeline_loaded, runtime_loaded)

    @classmethod
    def load_pickle_files(cls, pipeline_to_load: Pipeline, fitted_pipeline_id: str,
                          pipelines_fitted_dir: str, log_dir: str) -> Runtime:
        '''
            create a Runtime instance from given pipelines and supporting files
        '''

        runtime_loaded = Runtime(pipeline_to_load, fitted_pipeline_id=fitted_pipeline_id,
                                 volumes_dir=FittedPipeline.static_volume_dir, log_dir=log_dir)
        pipelines_fitted_dir = os.path.join(pipelines_fitted_dir, fitted_pipeline_id)
        for i, each_step in enumerate(pipeline_to_load.steps):
            # if it is a primitive, load directly
            if isinstance(each_step, PrimitiveStep):
                file_loc = os.path.join(pipelines_fitted_dir, "step_" + str(i) + ".pkl")
                with open(file_loc, "rb") as f:
                    each_step = pickle.load(f)
                    runtime_loaded.steps_state[i] = each_step

            # if it is a subpipeline, recursively creat a new runtime
            elif isinstance(each_step, SubpipelineStep):
                runtime_loaded.steps_state[i] = FittedPipeline.load_pickle_files(each_step.pipeline, log_dir, pipelines_fitted_dir)

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

        run = Runtime(state['pipeline'], fitted_pipeline_id=state['id'],
                      volumes_dir=FittedPipeline.static_volume_dir, log_dir=state['log_dir'])
        run.steps_state = fitted

        state['runtime'] = run

        self.__dict__ = state
