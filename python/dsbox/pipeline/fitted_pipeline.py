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
from d3m import utils as d3m_utils

from dsbox.template.runtime import Runtime,ForkedPdb
from dsbox.template.template import DSBoxTemplate

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
    # control parameters to let pipeline generator know whether we need add splitter
    need_splitter = False 
     # control parameters to let pipeline generator know whether we need add data mart related primitives
    need_data_augment = False

    def __init__(self, pipeline: Pipeline, dataset_id: str, log_dir: str, *, id: str = None,
                 metric_descriptions: typing.List = [], template: DSBoxTemplate = None,
                 template_name: str = None, template_task: str = None, template_subtask: str = None,
                 problem=None, extra_primitive = None) -> None:

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
        self.extra_primitive = extra_primitive
        self.log_dir = log_dir

        self.runtime = Runtime(pipeline, fitted_pipeline_id=self.id, template_name=self.template_name,
                               volumes_dir=FittedPipeline.static_volume_dir, log_dir=self.log_dir, task_type=self.template_task)

        self.metric_descriptions = list(metric_descriptions)
        self.runtime.set_metric_descriptions(self.metric_descriptions)

        self.metric: typing.Dict = {}
        self.auxiliary: typing.Dict = {}
        self._datamart_query_step_location = 0
        self.location_offset = 0
        _logger.debug('Creating fitted pipeline %s', self.id)

    def _set_fitted(self, fitted_pipe: typing.List[StepBase]) -> None:
        self.runtime.steps_state = fitted_pipe

    def set_metric(self, metric: typing.Dict):
        """
            Set the metric type for this fitted pipeline
        """
        self.metric = metric

    def get_primitive_augment(self, primitive_name:str , input_names:typing.List[str]) -> typing.Dict:
        """
            Base on the given primitive name and corresponding inputs_names
            Return the dict type primitive augment for adding in pipeline
        """
        if primitive_name == "splitter":
            from dsbox.datapreprocessing.cleaner.splitter import Splitter
            primitive_metadata = Splitter.metadata.query()

        elif primitive_name == "wikifier":
            from dsbox.datapreprocessing.cleaner.wikifier import Wikifier
            primitive_metadata = Wikifier.metadata.query()

        elif primitive_name == "denormalize":
            from common_primitives.denormalize import DenormalizePrimitive
            primitive_metadata = DenormalizePrimitive.metadata.query()

        elif "augment" in primitive_name:
            from common_primitives.datamart_augment import DataMartAugmentPrimitive
            primitive_metadata = DataMartAugmentPrimitive.metadata.query()

        primitive_augument= {
                              "type": "PRIMITIVE",
                              "primitive": {
                                "id": primitive_metadata['id'],
                                "version": primitive_metadata['version'],
                                "python_path": primitive_metadata['python_path'],
                                "name": primitive_metadata['name'],
                                "digest": primitive_metadata['digest']
                              },
                              "arguments": {
                                "inputs": {
                                  "type": "CONTAINER",
                                  "data": input_names[0] #"inputs.0"
                                }
                              },
                              "outputs": [
                                {
                                  "id": "produce"
                                }
                              ]
                            }

        # # special type of augment
        # if primitive_name == "datamart_augmentation":
        #     primitive_augument["arguments"] = {
        #                               "inputs1": {
        #                                   "type": "CONTAINER",
        #                                   "data":  input_names[0] #"steps."+str(self._datamart_query_step_location)+".produce"
        #                                 },
        #                                 "inputs2": {
        #                                   "type": "CONTAINER",
        #                                   "data":  input_names[1]#"inputs.0"
        #                                 }
        #                             }
        return primitive_augument


    def add_extra_primitive(self, primitive_name:typing.List[str], location_number:int) -> None:
        """
            Add extra primitives, usually it should be 
            "d3m.primitives.data_transformation.denormalize.Common"             or
            "d3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX"    or
            "d3m.primitives.data_augmentation.datamart_query.DSBOX"             or 
            "d3m.primitives.data_augmentation.datamart_augmentation.DSBOX"
        """
        structure = self.pipeline.to_json_structure()
        for each_primitive_name in primitive_name:
            # considering adding datamart query and augment must be add in the same time
            # we should support adding multiple primitives in once
            if location_number == 0:
                input_names = ["inputs.0"]
            else:
                input_names = ["steps."+str(location_number - 1)+".produce"]
            # if each_primitive_name == "datamart_augmentation":
            #     if location_number >= 2:
            #         input_names = ["steps."+str(location_number - 1)+".produce", "steps."+str(location_number - 2)+".produce"]
            #     if location_number == 1: # which should not occur any more
            #         _logger.warn("detect DatamartAugmentation primitive was added in second step, which should not happen!")
            #         input_names = ["steps."+str(location_number - 1)+".produce", "inputs.0"]

            primitive_augument = self.get_primitive_augment(each_primitive_name, input_names)

            hyperparams_file_loc = os.path.join(os.environ["D3MLOCALDIR"], self.dataset_id+each_primitive_name+".json")
            with open(hyperparams_file_loc, "r") as f:
                hyperparams_file = json.load(f)
            new_hyper_file = {}
            for key, value in hyperparams_file.items():
                new_hyper_file[key] = {"type":"VALUE",
                                       "data":value}
            primitive_augument['hyperparams'] =  new_hyper_file

            # update output reference
            output_step_reference = structure["outputs"] # it should look like "steps.11.produce"
            for i, each_output_step_reference in enumerate(output_step_reference):
                each_output_step_reference_split = each_output_step_reference["data"].split(".")
                each_output_step_reference_split[1] = str(int(each_output_step_reference_split[1]) + 1)
                structure["outputs"][i]["data"] = ".".join(each_output_step_reference_split)

            # add the step in corresponding position
            detail_steps = structure["steps"]
            detail_steps.insert(location_number, primitive_augument)
            for i in range(location_number+1, len(detail_steps)):
                each_step = detail_steps[i]
                if "arguments" in each_step:
                    for each_argument_key in each_step["arguments"].keys():
                        argument_target = each_step["arguments"][each_argument_key]["data"]
                        if argument_target == "inputs.0":# and "denormalize" in each_step["primitive"]["python_path"]:
                            argument_target_new = "steps.0.produce"
                            each_step["arguments"][each_argument_key]["data"] = argument_target_new
                        else:
                            argument_target_list = argument_target.split(".")
                            if int(argument_target_list[1]) >= location_number or i == location_number+1:
                                argument_target_list[1] = str(int(argument_target_list[1]) + 1)
                                argument_target_new = ".".join(argument_target_list)
                                each_step["arguments"][each_argument_key]["data"] = argument_target_new
                # update each_step
                detail_steps[i] = each_step
            # update original structure
            structure["steps"] = detail_steps
            # add into runtime
            primitive_pickle_file_loc = os.path.join(os.environ["D3MLOCALDIR"], self.dataset_id+each_primitive_name+".pkl")
            with open(primitive_pickle_file_loc, "rb") as f:
                primitive_pickle_file = pickle.load(f)
            self.runtime.steps_state.insert(location_number, primitive_pickle_file)
            location_number += 1

        # update cracked Pipeline from new structure
        self.pipeline = Pipeline.from_json_structure(structure)
        # ForkedPdb().set_trace()
        steps_state_old = self.runtime.steps_state
        # generate new runtime
        self.runtime = Runtime(self.pipeline, fitted_pipeline_id=self.id,
                                 volumes_dir=FittedPipeline.static_volume_dir, log_dir=self.log_dir)
        self.runtime.steps_state = steps_state_old
        

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
        # TODO: check is it always to be 0 here?
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
        if self.extra_primitive and "splitter" in self.extra_primitive:
            self.add_extra_primitive(["splitter"], self.location_offset)
            structure = self.pipeline.to_json_structure()
            _logger.info("Primitive Splitter has been added to pipeline.")
            self.location_offset += 1

        if self.extra_primitive and "denormalize" in self.extra_primitive:
            self.add_extra_primitive(["denormalize"], self.location_offset)
            structure = self.pipeline.to_json_structure()
            _logger.info("Primitive Denormalize has been added to pipeline.")
            self.location_offset += 1

        if self.extra_primitive and "wikifier" in self.extra_primitive:
            self.add_extra_primitive(["wikifier"], self.location_offset)
            structure = self.pipeline.to_json_structure()
            _logger.info("Primitive Wikifier has been added to pipeline.")
            self.location_offset += 1

        augment_count = 0
        current_augment = "augment" + str(augment_count)
        while current_augment in self.extra_primitive:
            self.add_extra_primitive([current_augment], self.location_offset)
            structure = self.pipeline.to_json_structure()
            _logger.info("Primitive " + current_augment +" has been added to pipeline.")
            augment_count += 1
            current_augment = "augment" + str(augment_count)

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
            if "values" in self.metric:
                structure['k_fold_validation_score'] = self.metric["values"]
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

        # update from d3m v2019.5.8: update digest to ensure the digest value is correct
        updated_digest = d3m_utils.compute_digest(Pipeline._canonical_pipeline_description(structure))
        structure['digest'] = updated_digest
        # save the pipeline with json format
        json_loc = os.path.join(pipeline_dir, self.pipeline.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out, separators=(',', ':'),indent=4)

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
                        json.dump(subpipeline_structure, out, separators=(',', ':'),indent=4)

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
