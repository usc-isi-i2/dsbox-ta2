import json
import logging
import os
import pickle
import pprint
import sys
import typing
import uuid

from d3m.metadata.pipeline import Pipeline, Resolver, StepBase, PrimitiveStep, SubpipelineStep
from d3m import exceptions
from d3m import utils as d3m_utils

from dsbox.controller.config import RuntimeSetting
from dsbox.template.runtime import Runtime
# from dsbox.template.runtime import ForkedPdb
from dsbox.template.template import DSBoxTemplate

from dsbox.schema import larger_is_better

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
    random_seed : int
        A random seed to use for every run. This control all randomness during the run.
    id: str
        the id of the pipeline
    metric_descriptions: List[str]
        describe the type of the metrics, needed for calculating the score of the predictions
    auxiliary: dict
        detail processes of fitted pipeline, not necessaary to be added
    """

    # Static volume for data need by d3m primitives
    # static_volume_dir: str = ""
    runtime_setting: RuntimeSetting = RuntimeSetting()

    # control parameters to let pipeline generator know whether we need add splitter
    need_splitter = False
    # control parameters to let pipeline generator know whether we need add data mart related primitives
    need_data_augment = False

    # D3M dirs
    pipelines_searched_subdir: str = 'pipelines_searched'
    pipelines_scored_subdir: str = 'pipelines_scored'
    pipelines_ranked_subdir: str = 'pipelines_ranked'
    subpipelines_subdir: str = 'subpipelines'
    pipeline_runs_subdir: str = 'pipeline_runs'

    # DSBox dirs
    pipelines_fitted_subdir: str = 'pipelines_fitted'
    pipelines_info_subdir: str = 'pipelines_info'

    def __init__(self, pipeline: Pipeline, dataset_id: str, *, id: str = None,
                 metric_descriptions: typing.List = [], template: DSBoxTemplate = None,
                 template_name: str = None, template_task: str = None, template_subtask: str = None,
                 problem=None, extra_primitive: typing.Set[str] = set(),
                 random_seed: int = -1) -> None:

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

        self.random_seed = random_seed
        self.runtime = Runtime(pipeline, fitted_pipeline_id=self.id, template_name=self.template_name,
                               task_type=self.template_task,
                               random_seed=self.random_seed,
                               volumes_dir=FittedPipeline.runtime_setting.volumes_dir,
                               log_dir=FittedPipeline.runtime_setting.log_dir)

        self.metric_descriptions = list(metric_descriptions)
        self.runtime.set_metric_descriptions(self.metric_descriptions)

        self.metric: typing.Dict = {}
        self.auxiliary: typing.Dict = {}
        self._datamart_query_step_location = 0
        self.location_offset = 0
        self.finished_add_extra_primitives = False

        _logger.debug('Creating fitted pipeline %s', self.id)

    def _set_fitted(self, fitted_pipe: typing.List[StepBase]) -> None:
        self.runtime.steps_state = fitted_pipe

    def set_metric(self, metric: typing.Dict):
        """
            Set the metric type for this fitted pipeline
        """
        self.metric = metric

    def get_primitive_augment(self, primitive_name: str, input_names: typing.List[str]) -> typing.Dict:
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

        primitive_augument = {
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
                                  "data": input_names[0]  # "inputs.0"
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

    def add_extra_primitive(self, primitive_name: typing.List[str], location_number: int) -> None:
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

            hyperparams_file_loc = os.path.join(self.runtime_setting.scratch_dir, self.dataset_id+each_primitive_name+".json")
            with open(hyperparams_file_loc, "r") as f:
                hyperparams_file = json.load(f)
            new_hyper_file = {}
            for key, value in hyperparams_file.items():
                new_hyper_file[key] = {"type": "VALUE",
                                       "data": value}
            primitive_augument['hyperparams'] = new_hyper_file

            # update output reference
            output_step_reference = structure["outputs"]  # it should look like "steps.11.produce"
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
                        if argument_target == "inputs.0":  # and "denormalize" in each_step["primitive"]["python_path"]:
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
            primitive_pickle_file_loc = os.path.join(self.runtime_setting.scratch_dir, self.dataset_id+each_primitive_name+".pkl")
            with open(primitive_pickle_file_loc, "rb") as f:
                primitive_pickle_file = pickle.load(f)
            self.runtime.steps_state.insert(location_number, primitive_pickle_file)
            location_number += 1

        # update cracked Pipeline from new structure
        self.pipeline = Pipeline.from_json_structure(structure)
        # ForkedPdb().set_trace()
        steps_state_old = self.runtime.steps_state
        # generate new runtime
        cross_validation_result = self.runtime.cross_validation_result
        self.runtime = Runtime(self.pipeline, fitted_pipeline_id=self.id,
                               random_seed=self.random_seed,
                               volumes_dir=FittedPipeline.runtime_setting.volumes_dir,
                               log_dir=FittedPipeline.runtime_setting.log_dir)
        self.runtime.cross_validation_result = cross_validation_result
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
        # return self.runtime.fit_outputs[step_number]
        # TODO: check is it always to be 0 here?
        # return self.runtime.fit_outputs[0]
        return self.runtime.fit_outputs.values['outputs.0']

    def get_produce_step_output(self, step_number: int):
        # return self.runtime.produce_outputs[step_number]
        # TODO: should here always be 0?
        # return self.runtime.produce_outputs[0]
        return self.runtime.produce_outputs.values['outputs.0']

    def save(self, folder_loc: str) -> None:
        # Save pipeline_run first, before add_extra_primitive() overwrite self.runtime
        self.save_pipeline_run(os.path.join(folder_loc, self.pipeline_runs_subdir))

        # D3M
        self.save_schema_only(folder_loc, self.pipelines_searched_subdir, subpipelines_subdir=self.subpipelines_subdir)
        self.save_schema_only(folder_loc, self.pipelines_scored_subdir)
        # Always save a ranked version. If there is a limit on the number of ranked pipelines, then the controller will
        # remove the extra ones.
        self.save_schema_only(folder_loc, self.pipelines_ranked_subdir)
        self.save_rank(os.path.join(folder_loc, self.pipelines_ranked_subdir))

        # DSBox
        self.save_pipeline_info(os.path.join(folder_loc, self.pipelines_info_subdir))
        self.save_fitted_pipeline(os.path.join(folder_loc, self.pipelines_fitted_subdir))

    def save_pipeline_run(self, folder_loc: str) -> None:
        '''
        Save pipeline_run
        '''
        saved = False
        for result in [self.runtime.fit_outputs, self.runtime.produce_outputs]:
            if result is None or result.pipeline_run is None:
                continue
            structure = result.pipeline_run.to_json_structure()
            filepath = os.path.join(folder_loc, structure['id'] + '.json')

            # Skip saving duplicate, since id is a hash of the content
            if os.path.exists(filepath):
                continue

            with open(filepath, 'w') as out:
                json.dump(structure, out)
                saved = True
        if saved:
            pipelines_dir = os.path.join(folder_loc, 'pipelines')
            if not os.path.exists(pipelines_dir):
                os.mkdir(pipelines_dir)
            pipeline_filepath = os.path.join(pipelines_dir, self.pipeline.id + '.json')
            if os.path.exists(pipeline_filepath):
                _logger.debug(f'Skipping pipeline for pipeline_run: {self.pipeline.id}')
            else:
                structure = self.pipeline.to_json_structure()
                with open(pipeline_filepath, 'w') as out:
                    json.dump(structure, out)


    def get_set_rank(self) -> float:
        rank = sys.float_info.max
        if self.metric:
            metric: str = self.metric['metric']
            value: float = self.metric['value']
            if larger_is_better(metric):
                if value > 0.0:
                    rank = 1 / value
                else:
                    rank = 10**5
            else:
                rank = value
            self.metric['rank'] = rank
        else:
            _logger.error("Metric type of the pipeline is unknown, unable to calculate the rank of the pipeline")
        return rank

    def save_rank(self, folder_loc: str):
        """
        Generate <pipeline_id>.rank file
        """
        with open(os.path.join(folder_loc, self.pipeline.id + '.rank'), 'w') as f:
            print(self.get_set_rank(), file=f)

    def save_schema_only(self, folder_loc: str, pipeline_schema_subdir: str, *, subpipelines_subdir: str = None) -> None:
        '''
        Save the pipline schema and subpline schema only.
        '''
        _logger.debug(f'Save pipeline schema {self.id}')

        pipeline_dir = os.path.join(folder_loc, pipeline_schema_subdir)

        structure = self.pipeline.to_json_structure()

        if not self.finished_add_extra_primitives:
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

            augment_count = -1
            for each in self.extra_primitive:
                if "augment" in each:
                    augment_count += 1

            current_augment = "augment" + str(augment_count)
            while current_augment in self.extra_primitive:
                self.add_extra_primitive([current_augment], self.location_offset)
                structure = self.pipeline.to_json_structure()
                _logger.info("Primitive " + current_augment + " has been added to pipeline.")
                augment_count -= 1
                current_augment = "augment" + str(augment_count)
            # no second time update
            self.finished_add_extra_primitives = True

        # update from d3m v2019.5.8: update digest to ensure the digest value is correct
        updated_digest = d3m_utils.compute_digest(Pipeline._canonical_pipeline_description(structure))
        structure['digest'] = updated_digest

        # save the pipeline with json format
        json_loc = os.path.join(pipeline_dir, self.pipeline.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out, separators=(',', ':'), indent=4)

        if subpipelines_subdir:
            subpipeline_dir = os.path.join(folder_loc, subpipelines_subdir)
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
                            json.dump(subpipeline_structure, out, separators=(',', ':'), indent=4)

    def save_fitted_pipeline(self, pipelines_fitted_dir: str) -> None:
        sub_dir = os.path.join(pipelines_fitted_dir, self.id)
        os.makedirs(sub_dir, exist_ok=True)

        # Save fitted pipeline structure
        structure = {
            'fitted_pipeline_id': self.id,
            'pipeline_id': self.pipeline.id,
            'dataset_id': self.dataset_id,
            'random_seed': self.random_seed
        }
        json_loc = os.path.join(pipelines_fitted_dir, self.id, self.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out)

        # pickle trained primitives
        assert len(self.runtime.steps_state) == len(self.pipeline.steps)
        self._save_pickle_files(pipelines_fitted_dir, self.runtime)

    def _save_pickle_files(self, pipelines_fitted_dir: str, input_runtime: Runtime) -> None:
        """
        Save fitted primitives
        """
        pipelines_fitted_dir = os.path.join(pipelines_fitted_dir, input_runtime.fitted_pipeline_id)
        os.makedirs(pipelines_fitted_dir, exist_ok=True)
        # save the pickle files of each primitive step
        for i in range(len(input_runtime.steps_state)):
            # print("Now saving step_", i)
            each_step = input_runtime.steps_state[i]
            if isinstance(each_step, Runtime):
                # if it is a subpipeline recursively call pickling functions
                self._save_pickle_files(pipelines_fitted_dir, each_step)
            else:
                # else if it is a primitive step, pickle directly
                file_loc = os.path.join(pipelines_fitted_dir, "step_" + str(i) + ".pkl")
                with open(file_loc, "wb") as f:
                    pickle.dump(each_step, f)

        _logger.info("Saving pickle files of pipeline {} finished.".format(self.id))

    def save_pipeline_info(self, folder_loc: str):
        """
        Save DSBox related information, including rank, cross_validation, task/subtask type, template.
        """
        structure: typing.Dict = {}
        # Save pipeline rank
        if self.metric:
            metric: str = self.metric['metric']
            value: float = self.metric['value']
            rank = self.get_set_rank()
            structure['pipeline_rank'] = rank
            structure['metric'] = metric.unparse()
            structure['metric_value'] = value
            if "values" in self.metric:
                structure['k_fold_validation_score'] = self.metric["values"]
        else:
            _logger.warn("Metric type of the pipeline is unknown, unable to calculate the rank of the pipeline")

        if self.runtime.cross_validation_result:
            structure['cross_validation'] = []
            for minfo in self.runtime.cross_validation_result:
                minfo['metric'] = minfo['metric'].unparse()
                structure['cross_validation'].append(minfo)

        structure['template_name'] = self.template_name
        structure['template_task'] = self.template_task
        structure['template_subtask'] = self.template_subtask

        if self.problem:
            structure['problem_taskType'] = self.problem['problem']['task_type'].unparse()
            try:
                structure['problem_taskSubType'] = self.problem['problem']['task_subtype'].unparse()
            except Exception:
                structure['problem_taskSubType'] = "NONE"
        else:
            _logger.warn("Problem type of the pipeline is unknown, unable to save problem taskType / taskSubtype")

        # save the pipeline with json format
        json_loc = os.path.join(folder_loc, self.pipeline.id + '.json')
        with open(json_loc, 'w') as out:
            json.dump(structure, out, indent=4)

    @classmethod
    def load_schema_only(cls, pipeline_id: str, folder_loc: str,
                         pipeline_schema_subdir: str) -> typing.Tuple[Pipeline, typing.Dict]:
        pipeline_dir = os.path.join(folder_loc, pipeline_schema_subdir)
        subpipeline_dir = os.path.join(folder_loc, cls.subpipelines_subdir)

        pipeline_schema = os.path.join(pipeline_dir, pipeline_id + '.json')

        with open(pipeline_schema, 'r') as f:
            structure = json.load(f)

        resolver = Resolver(pipeline_search_paths=[pipeline_dir, subpipeline_dir])
        pipeline = Pipeline.from_json_structure(pipeline_description=structure, resolver=resolver)
        return (pipeline, structure)

    @classmethod
    def load(cls, *, fitted_pipeline_id: str, folder_loc: str) -> 'FittedPipeline':
        # pipeline_schema_subdir: str = 'pipelines_scored', subpipelines_subdir: str = 'subpipelines', pipelines_fitted_subdir: str = 'pipelines_fitted'

        pipelines_fitted_dir = os.path.join(folder_loc, cls.pipelines_fitted_subdir)

        fitted_pipeline_schema = os.path.join(pipelines_fitted_dir, fitted_pipeline_id, fitted_pipeline_id+'.json')
        with open(fitted_pipeline_schema, 'r') as f:
            structure = json.load(f)

        pipeline, pipeline_structure = FittedPipeline.load_schema_only(structure['pipeline_id'], folder_loc, cls.pipelines_scored_subdir)
        runtime: Runtime = FittedPipeline.load_pickle_files(
            pipeline, structure['fitted_pipeline_id'], pipelines_fitted_dir=pipelines_fitted_dir)

        fitted_pipeline = FittedPipeline(pipeline, dataset_id=structure['dataset_id'], id=fitted_pipeline_id,
                                         random_seed=structure['dataset_id'])
        fitted_pipeline.runtime = runtime
        fitted_pipeline._set_fitted(runtime.steps_state)

        info_file = os.path.join(folder_loc, cls.pipelines_info_subdir, fitted_pipeline_id + '.json')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info = json.load(f)
            fitted_pipeline.template_name = info['template_name']
            fitted_pipeline.template_task = info['template_task']
            fitted_pipeline.template_subtask = info['template_subtask']
            if 'metric' in info:
                fitted_pipeline.metric = {
                    'metric': info['metric'],
                    'value': info['metric_value']
                }
            if 'cross_validation' in info:
                for metric_info in info['cross_validation']:
                    if 'metric' in metric_info:
                        metric_info['metric_info'] = PerformanceMetric.get_map()[metric_info['metric_info']]
                fitted_pipeline.runtime.cross_validation_result = info['cross_validation']
        return fitted_pipeline

    @classmethod
    def load_pickle_files(cls, pipeline_to_load: Pipeline, fitted_pipeline_id: str,
                          pipelines_fitted_dir: str) -> Runtime:
        '''
            create a Runtime instance from given pipelines and supporting files
        '''

        # What to set for random seed?
        random_seed = 0

        runtime_loaded = Runtime(pipeline_to_load, fitted_pipeline_id=fitted_pipeline_id,
                                 random_seed=random_seed,
                                 volumes_dir=FittedPipeline.runtime_setting.volumes_dir,
                                 log_dir=FittedPipeline.runtime_setting.log_dir)
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
                runtime_loaded.steps_state[i] = FittedPipeline.load_pickle_files(
                    each_step.pipeline, each_step.pipeline.id, pipelines_fitted_dir)

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
        state['id'] = self.id
        state['random_seed'] = self.random_seed
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
        random_seed = state['random_seed']

        run = Runtime(state['pipeline'], fitted_pipeline_id=state['id'],
                      random_seed=random_seed,
                      volumes_dir=FittedPipeline.runtime_setting.volumes_dir,
                      log_dir=FittedPipeline.runtime_setting.log_dir)
        run.steps_state = fitted

        state['runtime'] = run

        self.__dict__ = state
