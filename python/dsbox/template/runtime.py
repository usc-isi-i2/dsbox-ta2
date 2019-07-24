import contextlib
import logging
import pprint
import os
import sys
import tempfile
import time
import typing
import pdb

import numpy as np

import d3m.runtime as runtime_base

from collections import defaultdict
from multiprocessing import current_process

from pandas import DataFrame  # type: ignore
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore

from d3m import container
from d3m import exceptions
from d3m.metadata import base as metadata_base, pipeline as pipeline_module, pipeline_run as pipeline_run_module, problem
from d3m.primitive_interfaces import base

from dsbox.JobManager.cache import PrimitivesCache
from dsbox.template.utils import calculate_score, SpecialMetric

_logger = logging.getLogger(__name__)

MAX_DUMP_SIZE = 50  # 1000


class ForkedPdb(pdb.Pdb):
    """
    A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("theano").setLevel(logging.WARNING)
logging.getLogger("dill").setLevel(logging.WARNING)
logging.getLogger("d3m.metadata.pipeline_run").setLevel(logging.ERROR)


class Runtime(runtime_base.Runtime):
    """
    Class to run the build and run a Pipeline.

    Caution:
    Some method adapted from d3m's runtime, so if you find that our system can't run after
    updated the new d3m,
    It is extremely possible that d3m changed some of their codes on runtime and we copied part
    of their codes
    here in _run_primitive() function.

    Attributes
    ----------
    cache : PrimitivesCache
        Cache files used for primitive
    fit_outputs: d3m_container
        used to store the prediction outputs from fit() part's dataset
    fitted_pipeline_id : str
        A uuid format str to indicate the pipeline id
    random_seed : int
        A random seed to use for every run. This control all randomness during the run.
    log_dir : str
        A path to indicate the path to the logging files
    pipeline_description : Pipeline
        A pipeline description to be executed.
    produce_outputs: d3m_container
        used to store the prediction outputs from produce() part's dataset
    timing: dict{str : int}
        A dictionary used to store the time used for each step and total pipeline

    Parameters
    ----------
    fitted_pipeline_id : str
        A uuid format str to indicate the pipeline id
    log_dir : str
        A path to indicate the path to the logging files
    pipeline_description : Pipeline
        A pipeline description to be executed.
    """

    def __init__(
            self, pipeline: pipeline_module.Pipeline,  hyperparams: typing.Sequence = None, *,
            problem_description: problem.Problem = None, context: metadata_base.Context = metadata_base.Context.TESTING,
            random_seed: int = -1, volumes_dir: str = None, scratch_dir: str = None,
            is_standard_pipeline: bool = False, environment: pipeline_run_module.RuntimeEnvironment = None,
            users: typing.Sequence[pipeline_run_module.User] = None,
            # dsbox parameters
            fitted_pipeline_id: str = None, template_name: str = '', log_dir: str = None, task_type: str = '',
    ) -> None:
        if random_seed == -1:
            raise ValueError('Be sure to set random seed. Fix this.')

        super().__init__(
            pipeline=pipeline, hyperparams=hyperparams, problem_description=problem_description, context=context,
            random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
            is_standard_pipeline=is_standard_pipeline, environment=environment, users=users)
        # def __init__(self, pipeline_description: Pipeline, fitted_pipeline_id: str, log_dir) -> None:

        # super().__init__(pipeline=pipeline_description, hyperparams=None, problem_description=None)

        self.cache: PrimitivesCache = None
        self.cross_validation_result: typing.List = []
        if fitted_pipeline_id is None:
            # Occurs when runtime is not initialized by DSBox
            self.fitted_pipeline_id = pipeline.id
        else:
            self.fitted_pipeline_id = fitted_pipeline_id
        self.template_name = template_name
        self.fit_outputs: runtime_base.Result = None
        self.log_dir = log_dir
        self.metric_descriptions: typing.List[typing.Dict] = []
        self.produce_outputs = None
        self.timing: typing.Dict = {}
        self.timing["total_time_used"] = 0.0
        self.task_type = task_type
        # 2019-7-12: Not working turning cache off for now
        self.use_cache = False
        # self.timing["total_time_used_without_cache"] = 0.0

        # !
        self.skip_fit_phase = False

        # Debug mode. If true compare cache with actual result.
        self.validate_cache = True

    def set_not_use_cache(self) -> None:
        self.use_cache = False

    def set_metric_descriptions(self, metric_descriptions):
        self.metric_descriptions = metric_descriptions

    def _run_primitive(self, step: pipeline_module.PrimitiveStep) -> None:
        '''
            Override the d3m_runtime's function
            And add the cache support
        '''
        if step.primitive is None:
            raise exceptions.InvalidPipelineError("Primitive has not been resolved.")

        _logger.debug(f"Start primitive: {step.primitive.metadata.query()['name']}")

        primitive_base: typing.Type[base.PrimitiveBase] = step.primitive

        time_start = time.time()
        cache_hit: bool = False

        self.pipeline_run.add_primitive_step(step)
        arguments = self._prepare_primitive_arguments(step)

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            hyperparams = self._prepare_primitive_hyperparams(step)

            prim_name, prim_hash = self.cache._get_hash(
                hash_prefix=None,
                pipe_step=self.pipeline.steps[self.current_step],
                primitive_arguments=arguments,
                primitive_hyperparams=hyperparams
            )

            # Store cache_hit state. In parallel mode, cache may change state and cause primitive_actual not to be set.
            cache_hit = self.use_cache and self.cache.is_hit_key(prim_hash=prim_hash, prim_name=prim_name)

            if cache_hit:
                _logger.debug(f'Using cached primitive: {prim_name}, {prim_hash}')
                fitting_time, primitive = self.cache.lookup_key(prim_name=prim_name, prim_hash=prim_hash)
                if self.validate_cache:
                    primitive_actual = self._create_pipeline_primitive(step.primitive, hyperparams)
            else:
                # We create a primitive just before it is being fitted for the first time. This assures that any primitives
                # it depends on through its hyper-parameters have already been fitted (because they are in prior steps).
                # Similarly, any pipeline-based value being passed to a hyper-parameter has already been computed.
                primitive = self._create_pipeline_primitive(step.primitive, hyperparams)

            assert self.steps_state[self.current_step] is None
            self.steps_state[self.current_step] = primitive

        else:
            primitive = typing.cast(base.PrimitiveBase, self.steps_state[self.current_step])

            assert isinstance(primitive, base.PrimitiveBase), type(primitive)

        # If primitive step has no arguments we do not fit or produce it. It is meant to be used as
        # unfitted primitive for another primitive's hyper-parameter.
        if not arguments:
            return

        # Do cross validation
        do_cross_validation = '_dsbox_runtime' in step.__dict__ and "cross_validation" in step._dsbox_runtime
        if (do_cross_validation
            and self.phase == metadata_base.PipelineRunPhase.FIT
            and not cache_hit):

            hyperparams = self._prepare_primitive_hyperparams(step)
            fit_multi_produce_arguments = self._filter_arguments(step.primitive, 'fit_multi_produce', dict(arguments, produce_methods=step.outputs))
            multi_produce_arguments = self._filter_arguments(step.primitive, 'multi_produce', dict(arguments, produce_methods=step.outputs))
            self.cross_validation_result = self._cross_validation(
                primitive_base, hyperparams, fit_multi_produce_arguments, multi_produce_arguments, step._dsbox_runtime)

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            if cache_hit:
                fit_multi_produce_arguments = self._filter_arguments(step.primitive, 'multi_produce', dict(arguments, produce_methods=step.outputs))
                while True:
                    multi_call_result = self._call_primitive_method(primitive.multi_produce, fit_multi_produce_arguments)
                    if multi_call_result.has_finished:
                        outputs = multi_call_result.values
                        break

                if self.validate_cache:
                    fit_multi_produce_arguments = self._filter_arguments(step.primitive, 'fit_multi_produce', dict(arguments, produce_methods=step.outputs))
                    while True:
                        multi_call_result = self._call_primitive_method(primitive_actual.fit_multi_produce, fit_multi_produce_arguments)
                        if multi_call_result.has_finished:
                            outputs_actual = multi_call_result.values
                            break
                    is_equal, reason = self._equals(outputs_actual, outputs)
                    if not is_equal:
                        _logger.error(f'========== CACHED PRIMITIVE OUTPUT DIFFERS : {reason}')
                        _logger.error('==== Output from new created primtive')
                        _logger.error(pprint.pformat(outputs_actual))
                        _logger.error('==== Output from cached primitive')
                        _logger.error(pprint.pformat(outputs))

            else:
                # Primitve is newly create, must fit it
                fit_multi_produce_arguments = self._filter_arguments(step.primitive, 'fit_multi_produce', dict(arguments, produce_methods=step.outputs))

                # We fit and produce until fitting and producing finishes.
                # TODO: Support configuring limits on iterations/time.
                # TODO: Produce again only those produce methods which have not finished, currently we simply run all of them again.
                while True:
                    multi_call_result = self._call_primitive_method(primitive.fit_multi_produce, fit_multi_produce_arguments)
                    if multi_call_result.has_finished:
                        outputs = multi_call_result.values
                        break

                # Add fitted primitive to cache
                fitting_time = (time.time() - time_start)
                self.cache.push_key(prim_name=prim_name, prim_hash=prim_hash, model=primitive, fitting_time=fitting_time)

        elif self.phase == metadata_base.PipelineRunPhase.PRODUCE:
            multi_produce_arguments = self._filter_arguments(step.primitive, 'multi_produce', dict(arguments, produce_methods=step.outputs))

            # We produce until producing finishes.
            # TODO: Support configuring limits on iterations/time.
            # TODO: Produce again only those produce methods which have not finished, currently we simply run all of them again.
            while True:
                multi_call_result = self._call_primitive_method(primitive.multi_produce, multi_produce_arguments)
                if multi_call_result.has_finished:
                    outputs = multi_call_result.values
                    break
        else:
            # TODO: Allow dispatch to a general method so that subclasses of this class can handle them if necessary.
            raise exceptions.UnexpectedValueError("Unknown phase: {phase}".format(phase=self.phase))

        for output_id in step.outputs:
            output_data_reference = 'steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)

            if output_id in outputs:
                self.data_values[output_data_reference] = outputs[output_id]
            else:
                raise exceptions.InvalidReturnValueError("Missing declared output '{output_id}' in computed primitive's outputs.".format(output_id=output_id))
        if self.phase == metadata_base.PipelineRunPhase.FIT:
            self._log_step_output('fit', output_data_reference, step, self.data_values)
        else:
            self._log_step_output('pro', output_data_reference, step, self.data_values)

        # add up the timing
        self.timing["total_time_used"] += (time.time() - time_start)
        _logger.debug(f"Done  primitive: {step.primitive.metadata.query()['name']}")

    def _equals(self, outputs_actual: typing.Dict, outputs: typing.Dict) -> typing.Tuple[bool, str]:
        try:
            for key, actual in outputs_actual.items():
                if not key in outputs:
                    return False, f'Missing key {key}'
                if isinstance(actual, container.DataFrame):
                    if not actual.shape == outputs[key].shape:
                        return False, f'Dataframe shape different: {actual.shape} != {outputs[key].shape}'
                    for col in range(actual.shape[1]):
                        if actual.dtypes[col] == np.object_:
                            if not actual.iloc[:, col].equals(outputs[key].iloc[:, col]):
                                return False, f'column {col} does not match'
                        else:
                            if not np.allclose(actual.iloc[:, col], outputs[key].iloc[:, col], equal_nan=True):
                                return False, f'Column {col} of {actual.dtypes[col]} is not close'
                elif isinstance(actual, container.ndarray):
                    if not actual.shape == outputs[key].shape:
                        return False, f'Ndarray shape different: {actual.shape} != {outputs[key].shape}'
                    if not np.allclose(actual, outputs[key], equal_nan=True):
                        return False, f'Ndarray do not close'
                else:
                    if not actual==outputs[key]:
                        return False, f'Object not equal'
        except Exception as e:
            print('Exception: ', e)
        return True, ""

    # def _run_primitive_old(self, this_step: pipeline_module.PrimitiveStep) -> None:
    #     '''
    #         Override the d3m_runtime's function
    #         And add the cache support
    #     '''
    #     if this_step.primitive is None:
    #         raise exceptions.InvalidPipelineError("Primitive has not been resolved.")

    #     time_start = time.time()

    #     _logger.debug(f"running primitive: {this_step.primitive.metadata.query()['name']}")
    #     # call d3m's run primitive directly if not use cache
    #     # NOTE: But need to perform cross validation!
    #     if not self.use_cache:
    #         super()._run_primitive(this_step)

    #     elif self.phase == metadata_base.PipelineRunPhase.FIT:

    #         # Same as old codes, use argument as the cache system's key
    #         primitive_arguments = self._prepare_primitive_arguments(this_step)
    #         primitive_arguments["produce_methods"] = this_step.outputs
    #         prim_name, prim_hash = self.cache._get_hash(
    #             hash_prefix=None, pipe_step=self.pipeline.steps[self.current_step],
    #             primitive_arguments=primitive_arguments)

    #         if not self.skip_fit_phase:
    #             # if we need to do cross validation, do it before normal fit() step
    #             if '_dsbox_runtime' in this_step.__dict__ and "cross_validation" in this_step._dsbox_runtime:

    #                 primitive: typing.Type[base.PrimitiveBase] = this_step.primitive
    #                 # TODO: add one more "if" to restrict runtime to run cross validation only for tuning steps
    #                 primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #                 custom_hyperparams = dict()
    #                 # produce_params only have 'inputs'
    #                 produce_params = dict((k, primitive_arguments[k]) for k in ["inputs"])
    #                 # training_arguments have ['inputs', 'outputs']
    #                 training_arguments = dict((k, primitive_arguments[k]) for k in ["inputs","outputs"])

    #                 if bool(this_step.hyperparams):
    #                     for hyperparam, value in this_step.hyperparams.items():
    #                         if isinstance(value, dict):
    #                             custom_hyperparams[hyperparam] = value['data']
    #                         else:
    #                             custom_hyperparams[hyperparam] = value

    #                 # self.cross_validation_result = None
    #                 self.cross_validation_result = self._cross_validation(
    #                     primitive, training_arguments, produce_params, primitive_hyperparams,
    #                     custom_hyperparams, this_step._dsbox_runtime)

    #                 # print("!@#$$%$$$,cvfinished!!!")
    #                 # print(self.cross_validation_result)
    #             # END for cross-validation process

    #         cache_hit = False
    #         _logger.debug(
    #             "Primitive Fit. 'id': '%(primitive_id)s', '(name, hash)': ('%(name)s', "
    #             "'%(hash)s'), 'worker_id': '%(worker_id)s'.",
    #             {
    #                 'primitive_id':
    #                     self.pipeline.steps[self.current_step].get_primitive_id(),
    #                 'name': prim_name,
    #                 'hash': prim_hash,
    #                 'worker_id': current_process()
    #             },
    #         )

    #         # if this primitive hitted
    #         if self.cache.is_hit_key(prim_hash=prim_hash, prim_name=prim_name):
    #             # TODO: How to handle pipeline_run for cache hit?
    #             self.pipeline_run.add_primitive_step(this_step)

    #             fitting_time, model = self.cache.lookup_key(prim_name=prim_name, prim_hash=prim_hash)
    #             self.steps_state[self.current_step] = model

    #             # print cache reading time
    #             cache_reading_time = (time.time() - time_start)
    #             _logger.debug(f"[INFO] cache reading took {cache_reading_time} s and "
    #                           f"fitting time took {fitting_time} s")
    #             cache_hit = True
    #             # print("!!!!Fit step with hitted finished!!!!")

    #             # HERE the code adapted from d3m's runtime!!! If new version of runtime changd,
    #             # remember to change here
    #             # this part use the primitive adapted from cache to regenerate the prediction of
    #             # training dataset
    #             # and output the results to self.environment which is used to store the
    #             # intermediate results of each steps

    #             # === From v2018.7
    #             # multi_produce_arguments = self._filter_arguments(this_step.primitive,
    #             #                                                  'multi_produce',
    #             #                                                  dict(primitive_arguments,
    #             #                                                       produce_methods=this_step.outputs))
    #             # while True:
    #             #     multi_call_result = model.multi_produce(**multi_produce_arguments)
    #             #     if multi_call_result.has_finished:
    #             #         outputs = multi_call_result.values
    #             #         break
    #             # for output_id in this_step.outputs:
    #             #     output_data_reference = 'steps.{i}.{output_id}'.format(i=this_step.index,
    #             #                                                            output_id=output_id)
    #             #     self.environment[output_data_reference] = outputs[output_id]

    #             primitive = model
    #             # === From v2019.1.21
    #             fit_multi_produce_arguments = self._filter_arguments(this_step.primitive, 'fit_multi_produce', dict(primitive_arguments, produce_methods=this_step.outputs))
    #             while True:
    #                 multi_call_result = self._call_primitive_method(primitive.fit_multi_produce, fit_multi_produce_arguments)
    #                 if multi_call_result.has_finished:
    #                     outputs = multi_call_result.values
    #                     break

    #             for output_id in this_step.outputs:
    #                 output_data_reference = 'steps.{i}.{output_id}'.format(i=this_step.index, output_id=output_id)

    #                 if output_id in outputs:
    #                     self.data_values[output_data_reference] = outputs[output_id]
    #                 else:
    #                     raise exceptions.InvalidReturnValueError("Missing declared output '{output_id}' in computed primitive's outputs.".format(output_id=output_id))


    #             # if we did not find the cache, run the primitive with d3m's inner function
    #         else:
    #             # print("!!!!Fit step with not hit!!!!")
    #             super()._run_primitive(this_step)
    #             fitting_time = (time.time() - time_start)
    #             # get the model after fitting
    #             model = self.steps_state[self.current_step]
    #             # push the model to cache
    #             self.cache.push_key(prim_name=prim_name, prim_hash=prim_hash, model=model,
    #                                 fitting_time=fitting_time)

    #             self._check_primitive_output(this_step, self.data_values)

    #             # log fitting results
    #             for output_id in this_step.outputs:
    #                 output_data_reference = 'steps.{i}.{output_id}'.format(i=this_step.index, output_id=output_id)
    #             self._log_fitted_step(self.cache, output_data_reference, this_step, self.data_values)

    #         # END processing part for FIT Phase

    #     elif self.phase == metadata_base.PipelineRunPhase.PRODUCE:
    #         # if in produce step, always use the d3m's codes
    #         super()._run_primitive(this_step)

    #         for output_id in this_step.outputs:
    #             output_data_reference = 'steps.{i}.{output_id}'.format(i=this_step.index, output_id=output_id)
    #         self._log_produce_step(output_data_reference, this_step, self.data_values)

    #     else:
    #         raise exceptions.UnexpectedValueError("Unknown phase: {phase}".format(phase=self.phase))

    #     # add up the timing
    #     self.timing["total_time_used"] += (time.time() - time_start)
    #     _logger.debug(f"   done primitive: {this_step.primitive.metadata.query()['name']}")

    def _check_primitive_output(self, primitive_step, primitives_outputs):
        for output_id in primitive_step.outputs:
            output_data_reference = 'steps.{i}.{output_id}'.format(i=primitive_step.index, output_id=output_id)
            output = primitives_outputs[output_data_reference]
            if isinstance(output, DataFrame):
                row_size, col_size = primitives_outputs[output_data_reference].shape
                for col in range(col_size):
                    if len(output.metadata.query((metadata_base.ALL_ELEMENTS, col))) == 0:
                        _logger.warning(f'Incomplete metadata at col {col}. Primitive={primitive_step.primitive}')

    def _log_step_output(self, prefix: str, output_data_reference, primitive_step, primitives_outputs):
        '''
            The function use to record the intermediate output of each primitive and save into
            the logs

            Parameters
            ---------
            output_data_reference: a str use to indicate the dict key of the step output stored
            in self.environment
            primitive_step: indicate the primitive for logging
            primitives_outputs: the dict used to store outputs
        '''
        if _logger.getEffectiveLevel() <= 10:

            n_step = int(output_data_reference.split('.')[1])
            debug_file = os.path.join(
                self.log_dir, 'dfs',
                '{}_{}_{}_{}_{:02}_{}'.format(prefix, self.template_name, self.pipeline.id.split('-')[0],
                                              self.fitted_pipeline_id.split('-')[0], n_step,
                                              primitive_step.primitive))
            _logger.debug(
                "'template': '%(template)s', 'id': '%(pipeline_id)s', 'fitted': '%(fitted_pipeline_id)s', 'name': '%("
                "name)s', 'worker_id': '%(worker_id)s'. Output is written to: '%(path)s'.",
                {
                    'template': self.template_name,
                    'pipeline_id': self.pipeline.id,
                    'fitted_pipeline_id': self.fitted_pipeline_id,
                    'name': primitive_step.primitive,
                    'worker_id': current_process(),
                    'path': debug_file
                },
            )
            if primitives_outputs[output_data_reference] is None:
                with open(debug_file) as f:
                    f.write("None")
            else:
                if isinstance(primitives_outputs[output_data_reference], DataFrame):
                    try:
                        primitives_outputs[output_data_reference][:MAX_DUMP_SIZE].to_csv(debug_file)
                    except Exception:
                        pass
                try:
                    metadata_filepath = debug_file + '_meta'
                    with open(metadata_filepath, 'w') as out:
                        primitives_outputs[output_data_reference].metadata.pretty_print(handle=out)
                except Exception:
                    pass

    def _cross_validation(self,
                          primitive_base: typing.Type[base.PrimitiveBase],
                          hyperparams: typing.Dict,
                          fit_multi_produce_arguments: typing.Dict,
                          multi_produce_arguments: typing.Dict,
                          runtime_instr: typing.Dict,
                          seed: int = 4767
    ) -> typing.List:
        _logger.debug('cross-val primitive: %s' % str(primitive_base))

        results: typing.List[typing.Dict] = []

        validation_metrics: typing.Dict[str, typing.List[float]] = defaultdict(list)
        targets: typing.Dict[str, typing.List[list]] = defaultdict(list)

        X = fit_multi_produce_arguments['inputs']
        y = fit_multi_produce_arguments['outputs']

        cv = runtime_instr.get('cross_validation', 10)
        use_stratified = runtime_instr.get('stratified', False)

        # TODO: cross validation need to be update to fit with new requirement with adding indexes!!

        # Redirect stderr to an error file
        #  Directly assigning stderr to tempfile.TemporaryFile cause printing str to fail
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, str(primitive_base)), 'w') as errorfile:
                with contextlib.redirect_stderr(errorfile):

                    if use_stratified:
                        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
                    else:
                        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

                    try:
                        temp = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
                        for k, (train, test) in enumerate(temp.split(X, y)):
                            pass
                    except Exception as e:
                        _logger.error(f"Stratified failed, use KFold instead: {e}")
                        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

                    num = 0.0
                    for k, (train, test) in enumerate(kf.split(X, y)):
                        primitive = self._create_pipeline_primitive(primitive_base, hyperparams)

                        trainX = X.take(train, axis=0)
                        trainY = y.take(train, axis=0)#.values.ravel()
                        testX = X.take(test, axis=0)
                        testY = y.take(test, axis=0)#.values.ravel()

                        # reset index to be continuous
                        trainX = trainX.reset_index()
                        trainY = trainY.reset_index()
                        testX = testX.reset_index()
                        testY = testY.reset_index()
                        trainX = trainX.iloc[:, 1:]
                        trainY = trainY.iloc[:, 1:]
                        testX = testX.iloc[:, 1:]
                        testY = testY.iloc[:, 1:]

                        validation_train_arguments = dict(fit_multi_produce_arguments)
                        validation_train_arguments['inputs'] = trainX
                        validation_train_arguments['outputs'] = trainY

                        validation_test_arguments = dict(multi_produce_arguments)
                        validation_test_arguments['inputs'] = testX

                        try:
                            while True:
                                multi_call_result = self._call_primitive_method(primitive.fit_multi_produce, validation_train_arguments)
                                if multi_call_result.has_finished:
                                    train_outputs = multi_call_result.values
                                    break

                            while True:
                                multi_call_result = self._call_primitive_method(primitive.multi_produce, validation_test_arguments)
                                if multi_call_result.has_finished:
                                    test_outputs = multi_call_result.values
                                    break

                            ypred = test_outputs['produce']

                            num = num + 1.0

                            # if task type not given, take a guess
                            if self.task_type == "":
                                self._guess_task_type()

                            if 'd3mIndex' not in testY.columns:
                                testY.insert(0,'d3mIndex' ,testX['d3mIndex'].copy())
                            if 'd3mIndex' not in ypred.columns:
                                ypred.insert(0,'d3mIndex' ,testX['d3mIndex'].copy())

                            # update 2019.5.13: use calculate_score method instead from ConfigurationSpaceBaseSearch
                            # !!!! TODO: Use utils.score. How to fix this.
                            metric_score = calculate_score(
                                ground_truth=testY, prediction=ypred, performance_metrics=self.metric_descriptions,
                                task_type=self.task_type, regression_metric=SpecialMetric().regression_metric)

                            # targets['ground_truth'].append(testY)
                            # targets['prediction'].append(ypred)
                            for metric_description in metric_score:
                            #     metricDesc = problem.PerformanceMetric.parse(metric_description['metric'])
                            #     metric: typing.Callable = metricDesc.get_function()
                            #     params: typing.Dict = metric_description['params']
                                validation_metrics[metric_description['metric']].append(metric_description['value'])
                            # validation_metrics.append(metric_score)

                        except Exception as e:
                            sys.stderr.write(
                                "ERROR: cross_validation {}: {}\n".format(primitive_base, e))
                            # _logger.error("ERROR: cross_validation {}: {}\n".format(primitive_base, e))
                            _logger.exception("ERROR: cross_validation {}: {}\n".format(primitive_base, e))

        if num == 0:
            return results

        average_metrics: typing.Dict[str, dict] = {}
        for name, values in validation_metrics.items():
            if len(values) == 0:
                return results
            average_metrics[name] = sum(values) / len(values)

        for metric_description in validation_metrics:
            result_by_metric = {}
            result_by_metric['metric'] = metric_description
            result_by_metric['value'] = average_metrics[metric_description]
            result_by_metric['values'] = validation_metrics[metric_description]
            result_by_metric['targets'] = targets[metric_description]
            results.append(result_by_metric)

        for result in results:
            _logger.debug('cross-validation metric: %s=%.4f', result['metric'], result['value'])
            _logger.debug('cross-validation details: %s %s',
                          result['metric'], str(['%.4f' % x for x in result['values']]))

        return results

    # def _cross_validation_old(self, primitive: typing.Type[base.PrimitiveBase],
    #                       training_arguments: typing.Dict,
    #                       produce_params: typing.Dict,
    #                       primitive_hyperparams: hyperparams_module.Hyperparams,
    #                       custom_hyperparams: typing.Dict,
    #                       runtime_instr: typing.Dict,
    #                       seed: int = 4767) -> typing.List:

    #     _logger.debug('cross-val primitive: %s' % str(primitive))

    #     results: typing.List[typing.Dict] = []

    #     validation_metrics: typing.Dict[str, typing.List[float]] = defaultdict(list)
    #     targets: typing.Dict[str, typing.List[list]] = defaultdict(list)

    #     X = training_arguments['inputs']
    #     y = training_arguments['outputs']

    #     cv = runtime_instr.get('cross_validation', 10)
    #     use_stratified = runtime_instr.get('stratified', False)

    #     # TODO: cross validation need to be update to fit with new requirement with adding indexes!!

    #     # Redirect stderr to an error file
    #     #  Directly assigning stderr to tempfile.TemporaryFile cause printing str to fail
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         with open(os.path.join(tmpdir, str(primitive)), 'w') as errorfile:
    #             with contextlib.redirect_stderr(errorfile):

    #                 if use_stratified:
    #                     kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    #                 else:
    #                     kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    #                 try:
    #                     temp = kf
    #                     temp.__next__()
    #                 except:
    #                     _logger.error("Stratified failed, use KFold instead.")
    #                     kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    #                 num = 0.0
    #                 for k, (train, test) in enumerate(kf.split(X, y)):
    #                     try:
    #                         # !!!
    #                         # Temporary fix
    #                         # Still ignore the use_semantic types hyperparameters
    #                         if "use_semantic_types" in custom_hyperparams:
    #                             custom_hyperparams.pop("use_semantic_types")
    #                         if "return_result" in custom_hyperparams:
    #                             custom_hyperparams.pop("return_result")
    #                         if "add_index_columns" in custom_hyperparams:
    #                             custom_hyperparams.pop("add_index_columns")

    #                         model = primitive(hyperparams=primitive_hyperparams(
    #                             primitive_hyperparams.defaults(), **custom_hyperparams))
    #                     except Exception:
    #                         _logger.error(
    #                             "******************\n[ERROR]Hyperparameters unsuccesfully set - "
    #                             "using defaults")
    #                         model = primitive(
    #                             hyperparams=primitive_hyperparams(primitive_hyperparams.defaults()))

    #                     if model is None:
    #                         return results

    #                     trainX = X.take(train, axis=0)
    #                     trainY = y.take(train, axis=0)#.values.ravel()
    #                     testX = X.take(test, axis=0)
    #                     testY = y.take(test, axis=0)#.values.ravel()

    #                     # reset index to be continuous
    #                     trainX = trainX.reset_index()
    #                     trainY = trainY.reset_index()
    #                     testX = testX.reset_index()
    #                     testY = testY.reset_index()
    #                     trainX = trainX.iloc[:, 1:]
    #                     trainY = trainY.iloc[:, 1:]
    #                     testX = testX.iloc[:, 1:]
    #                     testY = testY.iloc[:, 1:]

    #                     validation_train = dict(training_arguments)
    #                     validation_train['inputs'] = trainX
    #                     validation_train['outputs'] = trainY

    #                     validation_test = dict(produce_params)
    #                     validation_test['inputs'] = testX

    #                     try:
    #                         model.set_training_data(**validation_train)
    #                         model.fit()
    #                         ypred = model.produce(**validation_test).value

    #                         num = num + 1.0

    #                         # if task type not given, take a guess
    #                         if self.task_type == "":
    #                             self._guess_task_type()

    #                         if 'd3mIndex' not in testY.columns:
    #                             testY.insert(0,'d3mIndex' ,testX['d3mIndex'].copy())
    #                         if 'd3mIndex' not in ypred.columns:
    #                             ypred.insert(0,'d3mIndex' ,testX['d3mIndex'].copy())

    #                         # update 2019.5.13: use calculate_score method instead from ConfigurationSpaceBaseSearch
    #                         # !!!! TODO: Use utils.score. How to fix this.
    #                         metric_score = calculate_score(
    #                             ground_truth=testY, prediction=ypred, performance_metrics=self.metric_descriptions,
    #                             task_type=self.task_type, regression_metric=SpecialMetric().regression_metric)

    #                         # targets['ground_truth'].append(testY)
    #                         # targets['prediction'].append(ypred)
    #                         for metric_description in metric_score:
    #                         #     metricDesc = problem.PerformanceMetric.parse(metric_description['metric'])
    #                         #     metric: typing.Callable = metricDesc.get_function()
    #                         #     params: typing.Dict = metric_description['params']
    #                             validation_metrics[metric_description['metric']].append(metric_description['value'])
    #                         # validation_metrics.append(metric_score)

    #                     except Exception as e:
    #                         sys.stderr.write(
    #                             "ERROR: cross_validation {}: {}\n".format(primitive, e))
    #                         _logger.error("ERROR: cross_validation {}: {}\n".format(primitive, e))
    #                         traceback.print_exc()

    #     if num == 0:
    #         return results

    #     average_metrics: typing.Dict[str, float] = {}
    #     for name, values in validation_metrics.items():
    #         if len(values) == 0:
    #             return results
    #         average_metrics[name] = sum(values) / len(values)

    #     for metric_description in validation_metrics:
    #         result_by_metric = {}
    #         result_by_metric['metric'] = metric_description
    #         result_by_metric['value'] = average_metrics[metric_description]
    #         result_by_metric['values'] = validation_metrics[metric_description]
    #         result_by_metric['targets'] = targets[metric_description]
    #         results.append(result_by_metric)

    #     for result in results:
    #         _logger.debug('cross-validation metric: %s=%.4f', result['metric'], result['value'])
    #         _logger.debug('cross-validation details: %s %s',
    #                       result['metric'], str(['%.4f' % x for x in result['values']]))

    #     return results

    def _guess_task_type(self):
        for each in self.metric_descriptions:
            if 'error' in each['metric'].unparse():
                self.task_type = "REGRESSION"
            else:
                self.task_type = "CLASSIFICATION"

    def fit(self, inputs: typing.Sequence[typing.Any], **arguments: typing.Any) -> runtime_base.Result:
        """
        Train all steps in the pipeline.

        Paramters
        ---------
        arguments
            Arguments required to train the Pipeline
        """
        if 'cache' in arguments:
            _logger.debug("Using global cache")
            self.cache = arguments['cache']
        else:
            _logger.debug("Using local cache")
            self.cache = PrimitivesCache()

        self.fit_outputs = super().fit(inputs=inputs)
        self.check_results(self.fit_outputs)
        return self.fit_outputs

    def produce(self, inputs: typing.Sequence[typing.Any], **arguments: typing.Any) -> runtime_base.Result:
        """
        Train all steps in the pipeline.

        Paramters
        ---------
        arguments
            Arguments required to execute the Pipeline
        """

        self.produce_outputs = super().produce(inputs=inputs)
        self.check_results(self.produce_outputs)
        return self.produce_outputs

    def check_results(self, res: runtime_base.Result):
        if res.error is None:
            assert len(res.values) > 0
        else:
            raise res.error

# Command-line related code copied from D3M runtime.py
import argparse
import json
import frozendict  # type: ignore
import pandas
import pickle
import uuid
from urllib import parse as url_parse
from pathlib import Path

from d3m import deprecate, utils
from d3m.container import dataset as dataset_module
from d3m.container import utils as container_utils

logger = logging.getLogger(__name__)

# dsbox
from d3m.runtime import Result, MultiResult

DEFAULT_SCORING_PIPELINE_ID = 'f596cd77-25f8-4d4c-a350-bb30ab1e58f6'
DEFAULT_SCORING_PIPELINE_PATH = os.path.join(
    os.path.dirname(runtime_base.__file__), 'contrib', 'pipelines', DEFAULT_SCORING_PIPELINE_ID + '.yml',
)


def _prepare_hyperparams(free_hyperparams: typing.Sequence, hyperparameter_values: typing.Dict) -> typing.Tuple[typing.Sequence, typing.Set[str]]:
    """
    Values in ``hyperparameter_values`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    hyperparams: typing.List[typing.Union[typing.Dict, typing.Sequence]] = []

    hyperparameter_values_used = set()

    for free_hyperparams_for_step in free_hyperparams:
        if isinstance(free_hyperparams_for_step, (dict, frozendict.frozendict)):
            values = {}
            for name, hyperparameter in free_hyperparams_for_step.items():
                if name in hyperparameter_values:
                    values[name] = hyperparameter.value_from_json_structure(json.loads(hyperparameter_values[name]))
                    hyperparameter_values_used.add(name)
            hyperparams.append(values)
        elif utils.is_sequence(free_hyperparams_for_step):
            step_hyperparams, step_hyperparameter_values_used = _prepare_hyperparams(free_hyperparams_for_step, hyperparameter_values)
            hyperparams.append(step_hyperparams)
            hyperparameter_values_used.update(step_hyperparameter_values_used)
        else:
            raise exceptions.UnexpectedValueError("Unknown hyper-parameters type: {hyperparams_type}".format(hyperparams_type=type(free_hyperparams_for_step)))

    return hyperparams, hyperparameter_values_used


# TODO: Add debug logging.
def fit(
    pipeline: pipeline_module.Pipeline, problem_description: typing.Optional[problem.Problem],
    inputs: typing.Sequence[container.Dataset], *, context: metadata_base.Context, hyperparams: typing.Sequence = None,
    random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None,
    runtime_environment: pipeline_run_module.RuntimeEnvironment = None, is_standard_pipeline: bool = True,
    expose_produced_outputs: bool = False,
) -> typing.Tuple[typing.Optional[Runtime], typing.Optional[container.DataFrame], Result]:
    for input in inputs:
        if not isinstance(input, container.Dataset):
            raise TypeError("A standard pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(input),
            ))

    if is_standard_pipeline and len(pipeline.outputs) != 1:
        raise ValueError("A standard pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(pipeline.outputs),
        ))

    runtime = Runtime(
        pipeline, hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
        is_standard_pipeline=is_standard_pipeline, environment=runtime_environment,
        # dsbox
        log_dir=os.environ['DSBOX_LOGGING_DIR'],
    )

    if expose_produced_outputs:
        return_values = None
    else:
        return_values = ['outputs.0']

    result = runtime.fit(inputs, return_values=return_values)

    if result.has_error():
        return None, None, result

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A standard pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    return runtime, output, result


# TODO: Add debug logging.
def produce(
    fitted_pipeline: Runtime, test_inputs: typing.Sequence[container.Dataset], *,
    expose_produced_outputs: bool = False,
) -> typing.Tuple[typing.Optional[container.DataFrame], Result]:
    for test_input in test_inputs:
        if not isinstance(test_input, container.Dataset):
            raise TypeError("A standard pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(test_input),
            ))

    # This is checked in "fit" already, but maybe somebody fitter a pipeline not through "fit".
    if fitted_pipeline.is_standard_pipeline and len(fitted_pipeline.pipeline.outputs) != 1:
        raise ValueError("A standard pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(fitted_pipeline.pipeline.outputs),
        ))

    if expose_produced_outputs:
        return_values = None
    else:
        return_values = ['outputs.0']

    result = fitted_pipeline.produce(test_inputs, return_values=return_values)
    if result.has_error():
        return None, result

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A standard pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    return output, result


# TODO: Add debug logging.
def score(
    scoring_pipeline: pipeline_module.Pipeline, problem_description: typing.Optional[problem.Problem], predictions: container.DataFrame,
    score_inputs: typing.Sequence[container.Dataset], metrics: typing.Sequence[typing.Dict], predictions_random_seed: int = None, *,
    context: metadata_base.Context, scoring_params: typing.Dict[str, str] = None, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.Optional[container.DataFrame], Result]:
    for score_input in score_inputs:
        if not isinstance(score_input, container.Dataset):
            raise TypeError("A scoring pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(score_input),
            ))

    if len(scoring_pipeline.outputs) != 1:
        raise ValueError("A scoring pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(scoring_pipeline.outputs),
        ))

    metrics_hyperparameter = []
    for metric in metrics:
        # Structure should match what "value_from_json_structure" would
        # return for "ComputeScoresPrimitive" hyper-parameter.
        # TODO: Once "ComputeScoresPrimitive" is moved to core package, use its default hyper-parameters here.
        metric_hyperparameter = {'metric': metric['metric'].name, 'k': None, 'pos_label': None}
        metric_hyperparameter.update(metric.get('params', {}))
        metrics_hyperparameter.append(metric_hyperparameter)

    if scoring_params is None:
        scoring_params = {}

    if metrics_hyperparameter:
        # We have to JSON-serialize it because "_prepare_hyperparams" expects
        # all values to be JSON-serialized.
        scoring_params['metrics'] = json.dumps(metrics_hyperparameter)

    hyperparams, scoring_params_used = _prepare_hyperparams(scoring_pipeline.get_free_hyperparams(), scoring_params)

    scoring_params_keys_set = set(scoring_params.keys())
    if scoring_params_keys_set - scoring_params_used:
        logger.warning("Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s", {
            'pipeline_id': scoring_pipeline.id,
            'unused_params': ', '.join(sorted(scoring_params_keys_set - scoring_params_used)),
        })

    runtime = Runtime(
        scoring_pipeline, hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
        environment=runtime_environment,
        # dsbox
        log_dir=os.environ['DSBOX_LOGGING_DIR'],
    )

    inputs = [predictions] + list(score_inputs)  # type: ignore

    # Fit + produce on same data.
    result = runtime.fit(inputs, return_values=['outputs.0'])
    if result.has_error():
        return None, result

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A scoring pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    if predictions_random_seed is not None:
        output = combine_random_seed(output, predictions_random_seed)

    return output, result


# TODO: Add debug logging.
def prepare_data(
    data_pipeline: pipeline_module.Pipeline, problem_description: typing.Optional[problem.Problem], inputs: typing.Sequence[container.Dataset],
    data_params: typing.Dict[str, str], *, context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.List, Result]:
    """
    Values in ``data_params`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    for input in inputs:
        if not isinstance(input, container.Dataset):
            raise TypeError("A data preparation pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(input),
            ))

    if len(data_pipeline.outputs) != 3:
        raise ValueError("A data preparation pipeline should have exactly three outputs, not {outputs}.".format(
            outputs=len(data_pipeline.outputs),
        ))

    if 'number_of_folds' in data_params:
        number_of_folds = int(data_params['number_of_folds'])
    else:
        # For now we assume other data preparation pipelines do only one fold. We should standardize
        # more hyper-parameters to gather how many folds have to be made (and not really folds, but
        # more how many input indices have to be passed to the pipeline).
        number_of_folds = 1

    data_hyperparams, data_params_used = _prepare_hyperparams(data_pipeline.get_free_hyperparams(), data_params)

    data_params_keys_set = set(data_params.keys())
    if data_params_keys_set - data_params_used:
        logger.warning("Not all provided hyper-parameters for the data preparation pipeline {pipeline_id} were used: {unused_params}".format(
            pipeline_id=data_pipeline.id,
            unused_params=sorted(data_params_keys_set - data_params_used),
        ))

    runtime = Runtime(
        data_pipeline, data_hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir,
        scratch_dir=scratch_dir, environment=runtime_environment,
        # dsbox
        log_dir=os.environ['DSBOX_LOGGING_DIR'],
    )

    # Fit + produce on same data. The inputs are the list of indices of folds
    # to generate and a dataset to split.
    result = runtime.fit([container.List(range(number_of_folds))] + list(inputs), return_values=['outputs.0', 'outputs.1', 'outputs.2'])  # type: ignore
    if result.has_error():
        return [], result

    outputs = [result.values['outputs.0'], result.values['outputs.1'], result.values['outputs.2']]

    for output in outputs:
        if not isinstance(output, container.List):
            raise TypeError("A data preparation pipeline's output should be of a container List type, not {input_type}.".format(
                input_type=type(output),
            ))
        if len(output) != number_of_folds:
            raise ValueError("A data preparation pipeline's output should contain {number_of_folds} datasets, not {length}.".format(
                number_of_folds=number_of_folds,
                length=len(output),
            ))

    return outputs, result


# TODO: Add debug logging.
def evaluate(
    pipeline: pipeline_module.Pipeline, data_pipeline: pipeline_module.Pipeline,
    scoring_pipeline: pipeline_module.Pipeline, problem_description: typing.Optional[problem.Problem],
    inputs: typing.Sequence[container.Dataset], data_params: typing.Dict[str, str],
    metrics: typing.Sequence[typing.Dict], *, context: metadata_base.Context,
    scoring_params: typing.Dict[str, str] = None, hyperparams: typing.Sequence = None, random_seed: int = 0,
    data_random_seed: int = 0, scoring_random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.List[container.DataFrame], MultiResult]:
    """
    Values in ``data_params`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    outputs, data_result = prepare_data(
        data_pipeline, problem_description, inputs, data_params,
        context=context, random_seed=data_random_seed, volumes_dir=volumes_dir,
        scratch_dir=scratch_dir, runtime_environment=runtime_environment,
    )
    if data_result.has_error():
        return [], MultiResult([data_result])

    fold_group_uuid = uuid.uuid4()

    all_scores: typing.List[container.DataFrame] = []
    all_results = MultiResult()
    for fold_index, (train_inputs, test_inputs, score_inputs) in enumerate(zip(*outputs)):
        fitted_pipeline, predictions, fit_result = fit(
            pipeline, problem_description, [train_inputs], context=context, hyperparams=hyperparams,
            random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
            runtime_environment=runtime_environment,
        )

        # Modifies "fit_result.pipeline_run" in-place.
        combine_pipeline_runs(
            fit_result.pipeline_run, data_pipeline_run=data_result.pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index,
        )

        all_results.append(fit_result)
        if fit_result.has_error():
            assert all_results.has_error()
            return all_scores, all_results

        predictions, produce_result = produce(fitted_pipeline, [test_inputs])

        # Modifies "produce_result.pipeline_run" in-place.
        combine_pipeline_runs(
            produce_result.pipeline_run, data_pipeline_run=data_result.pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index
        )

        all_results.append(produce_result)
        if produce_result.has_error():
            assert all_results.has_error()
            return all_scores, all_results

        scores, score_result = score(
            scoring_pipeline, problem_description, predictions, [score_inputs], metrics, random_seed,
            scoring_params=scoring_params, context=context, random_seed=scoring_random_seed, volumes_dir=volumes_dir,
            scratch_dir=scratch_dir, runtime_environment=runtime_environment,
        )

        # Modifies "produce_result.pipeline_run" in-place.
        combine_pipeline_runs(
            produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run,
        )
        # Sets the error, if there are any.
        produce_result.error = score_result.error

        # We modified "produce_result.pipeline_run" in-place and "produce_result"
        # is already among "all_results", so we do not add it again.
        if score_result.has_error():
            assert all_results.has_error()
            return all_scores, all_results

        # Modifies "produce_result.pipeline_run" in-place.
        combine_pipeline_runs(
            produce_result.pipeline_run, metrics=metrics, scores=scores,
        )

        all_scores.append(scores)

    return all_scores, all_results


is_uri = deprecate.function(message="use d3m.utils.is_uri instead")(utils.is_uri)

get_dataset = deprecate.function(message="use d3m.container.dataset.get_dataset instead")(dataset_module.get_dataset)
get_problem = deprecate.function(message="use d3m.metadata.problem.get_problem instead")(problem.get_problem)
get_pipeline = deprecate.function(message="use d3m.metadata.pipeline.get_pipeline instead")(pipeline_module.get_pipeline)


# TODO: Do not traverse the datasets directory every time.
def parse_meta(
    meta_file: typing.TextIO, datasets_dir: str, *, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None, compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING,
    strict_digest: bool = False, handle_score_split: bool = True,
) -> typing.Dict:
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    if datasets_dir is None:
        raise exceptions.InvalidArgumentValueError("Dataset directory has to be provided to resolve meta files.")

    meta = json.load(meta_file)

    datasets: typing.Dict[str, str] = {}
    problem_descriptions: typing.Dict[str, str] = {}

    for dirpath, dirnames, filenames in os.walk(datasets_dir, followlinks=True):
        if 'datasetDoc.json' in filenames:
            # Do not traverse further (to not parse "datasetDoc.json" or "problemDoc.json" if they
            # exists in raw data filename).
            dirnames[:] = []

            dataset_path = os.path.join(os.path.abspath(dirpath), 'datasetDoc.json')

            try:
                with open(dataset_path, 'r', encoding='utf8') as dataset_file:
                    dataset_doc = json.load(dataset_file)

                dataset_id = dataset_doc['about']['datasetID']
                # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
                # They are the same as TEST dataset splits, but we present them differently, so that
                # SCORE dataset splits have targets as part of data. Because of this we also update
                # corresponding dataset ID.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
                if handle_score_split and os.path.exists(os.path.join(dirpath, '..', 'targets.csv')) and dataset_id.endswith('_TEST'):
                    dataset_id = dataset_id[:-5] + '_SCORE'

                if dataset_id in datasets:
                    logger.warning(
                        "Duplicate dataset ID '%(dataset_id)s': '%(old_dataset)s' and '%(dataset)s'", {
                            'dataset_id': dataset_id,
                            'dataset': dataset_path,
                            'old_dataset': datasets[dataset_id],
                        },
                    )
                else:
                    datasets[dataset_id] = dataset_path

            except (ValueError, KeyError):
                logger.exception(
                    "Unable to read dataset '%(dataset)s'.", {
                        'dataset': dataset_path,
                    },
                )

        if 'problemDoc.json' in filenames:
            # We continue traversing further in this case.

            problem_path = os.path.join(os.path.abspath(dirpath), 'problemDoc.json')

            try:
                with open(problem_path, 'r', encoding='utf8') as problem_file:
                    problem_doc = json.load(problem_file)

                problem_id = problem_doc['about']['problemID']
                # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
                # They are the same as TEST dataset splits, but we present them differently, so that
                # SCORE dataset splits have targets as part of data. Because of this we also update
                # corresponding problem ID.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
                if handle_score_split and os.path.exists(os.path.join(dirpath, '..', 'targets.csv')) and problem_id.endswith('_TEST'):
                    problem_id = problem_id[:-5] + '_SCORE'

                    # Also update dataset references.
                    for data in problem_doc.get('inputs', {}).get('data', []):
                        if data['datasetID'].endswith('_TEST'):
                            data['datasetID'] = data['datasetID'][:-5] + '_SCORE'

                if problem_id in problem_descriptions:
                    logger.warning(
                        "Duplicate problem ID '%(problem_id)s': '%(old_problem)s' and '%(problem)s'", {
                            'problem_id': problem_id,
                            'problem': problem_path,
                            'old_problem': problem_descriptions[problem_id],
                        },
                    )
                else:
                    problem_descriptions[problem_id] = problem_path

            except (ValueError, KeyError):
                logger.exception(
                    "Unable to read problem description '%(problem)s'.", {
                        'problem': problem_path,
                    },
                )

    return {
        'problem': problem_resolver(problem_descriptions[meta['problem']]),
        'full_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['full_inputs']],
        'train_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['train_inputs']],
        'test_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['test_inputs']],
        'score_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['score_inputs']],
    }


def combine_random_seed(scores: container.DataFrame, random_seed: int) -> container.DataFrame:
    random_seed_column = container.DataFrame({'randomSeed': [random_seed] * scores.shape[0]})
    # We add the new column at the end so that we do not have to do complicated changes to the metadata.
    output_scores = pandas.concat([scores, random_seed_column], axis=1)
    # There is one more column now, so we update metadata for it.
    output_scores.metadata = scores.metadata.update((metadata_base.ALL_ELEMENTS,), {
        'dimension': {
            'length': output_scores.shape[1],
        },
    })
    output_scores.metadata = output_scores.metadata.update_column(output_scores.shape[1] - 1, {
        'name': 'randomSeed',
        'structural_type': int,
    })

    return output_scores


def combine_folds(scores_list: typing.List[container.DataFrame]) -> container.DataFrame:
    # We combine multiple scores tables into one output table by adding a "fold" column.
    for fold, scores in enumerate(scores_list):
        fold_column = container.DataFrame({'fold': [fold] * scores.shape[0]})
        # We add the new column at the end so that we do not have to do complicated
        # changes to the metadata.
        scores_list[fold] = pandas.concat([scores, fold_column], axis=1)
        # There is one more column now, so we update metadata for it.
        scores_list[fold].metadata = scores.metadata.update((metadata_base.ALL_ELEMENTS,), {
            'dimension': {
                'length': scores_list[fold].shape[1],
            },
        })
        scores_list[fold].metadata = scores_list[fold].metadata.update_column(scores_list[fold].shape[1] - 1, {
            'name': 'fold',
            'structural_type': int,
        })

    scores = pandas.concat(scores_list, axis=0).reset_index(drop=True)
    # We reuse metadata from the first fold and update the number of rows which is now
    # combined across all folds.
    scores.metadata = scores_list[0].metadata.update((), {
        'dimension': {
            'length': scores.shape[0],
        },
    })

    return scores


def combine_pipeline_runs(
    standard_pipeline_run: pipeline_run_module.PipelineRun, *,
    data_pipeline_run: pipeline_run_module.PipelineRun = None, scoring_pipeline_run: pipeline_run_module.PipelineRun = None,
    score_inputs: typing.Sequence[typing.Any] = None, metrics: typing.Sequence[typing.Dict] = None, scores: container.DataFrame = None,
    fold_group_uuid: uuid.UUID = None, fold_index: int = None,
) -> None:
    fold_args_provided = (item is None for item in (fold_group_uuid, fold_index))
    if any(fold_args_provided) and not all(fold_args_provided):
        raise exceptions.InvalidArgumentValueError("If any of 'fold_group_uuid' and 'fold_index' are provided, they must all be provided.")

    scores_args_provided = (item is None for item in (scores, metrics))
    if any(scores_args_provided) and not all(scores_args_provided):
        raise exceptions.InvalidArgumentValueError("If any of 'scores' or 'metrics' is provided, they must both be provided.")

    if data_pipeline_run is not None:
        standard_pipeline_run.set_data_preparation_pipeline_run(data_pipeline_run)

    if fold_group_uuid is not None:
        standard_pipeline_run.set_fold_group(fold_group_uuid, fold_index)

    if scoring_pipeline_run is not None:
        standard_pipeline_run.set_scoring_pipeline_run(scoring_pipeline_run, score_inputs)

    if scores is not None:
        standard_pipeline_run.set_scores(scores, metrics)


@deprecate.function(message="use extended DataFrame.to_csv method instead")
def export_dataframe(dataframe: container.DataFrame, output_file: typing.TextIO = None) -> typing.Optional[str]:
    return dataframe.to_csv(output_file)


def _check_duplicate_metrics(metrics: typing.Sequence[typing.Dict]) -> None:
    """
    In results from scoring we identify each score by its metric name. So to map those rows in scoring
    output back to requested metrics, names must be unique. Otherwise we would not know to which
    metric configuration the score belongs to.
    """

    only_metrics = [metric['metric'] for metric in metrics]

    if utils.has_duplicates(only_metrics):
        raise exceptions.InvalidArgumentValueError("Same metric listed multiple times.")


def get_metrics_from_list(metrics: typing.Sequence[str]) -> typing.Sequence[typing.Dict]:
    metric_descriptions = [{'metric': problem.PerformanceMetric[metric]} for metric in metrics]

    _check_duplicate_metrics(metric_descriptions)

    return metric_descriptions


def get_metrics_from_problem_description(problem_description: typing.Optional[problem.Problem]) -> typing.Sequence[typing.Dict]:
    if problem_description is None:
        return []

    metric_descriptions = problem_description['problem'].get('performance_metrics', [])

    _check_duplicate_metrics(metric_descriptions)

    return metric_descriptions


def _output_pipeline_runs(arguments: argparse.Namespace, pipeline_runs: typing.Sequence[pipeline_run_module.PipelineRun]) -> None:
    if not getattr(arguments, 'output_run', None):
        return

    first = True
    for pipeline_run in pipeline_runs:
        pipeline_run.to_yaml(arguments.output_run, appending=not first)
        first = False


def fit_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    pipeline = pipeline_resolver(
        arguments.pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None),
            compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['train_inputs']
    else:
        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem)
        else:
            problem_description = None

        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]

    is_standard_pipeline = getattr(arguments, 'standard_pipeline', True)

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    fitted_pipeline, predictions, result = fit(
        pipeline, problem_description, inputs,
        context=context, random_seed=getattr(arguments, 'random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
        is_standard_pipeline=is_standard_pipeline,
        expose_produced_outputs=expose_produced_outputs,
    )

    _output_pipeline_runs(arguments, [result.pipeline_run])

    result.check_success()

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    if getattr(arguments, 'output', None) is not None:
        assert is_standard_pipeline
        predictions.to_csv(arguments.output)

    if expose_produced_outputs:
        save_steps_outputs(result, arguments.expose_produced_outputs_dir)


# We have "pipeline_resolver" and "problem_resolver" as arguments (even if we are not
# using them in this function) so that the signature is the same for all handlers.
def produce_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset

    fitted_pipeline = pickle.load(arguments.fitted_pipeline)

    if not fitted_pipeline.is_standard_pipeline and getattr(arguments, 'output', None) is not None:
        raise exceptions.InvalidArgumentValueError("You cannot save predictions for a non-standard pipeline.")

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None),
            compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        test_inputs = meta['test_inputs']
    else:
        test_inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'test_inputs', [])
        ]

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    predictions, result = produce(fitted_pipeline, test_inputs, expose_produced_outputs=expose_produced_outputs)

    _output_pipeline_runs(arguments, [result.pipeline_run])

    result.check_success()

    if getattr(arguments, 'output', None) is not None:
        assert fitted_pipeline.is_standard_pipeline
        predictions.to_csv(arguments.output)

    if expose_produced_outputs:
        save_steps_outputs(result, arguments.expose_produced_outputs_dir)


# We have "problem_resolver" as an arguments (even if we are not
# using it in this function) so that the signature is the same for all handlers.
def score_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    fitted_pipeline = pickle.load(arguments.fitted_pipeline)
    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        test_inputs = meta['test_inputs']
        score_inputs = meta['score_inputs']
    else:
        test_inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'test_inputs', [])
        ]
        score_inputs = [
            dataset_resolver(
                score_input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for score_input_uri in getattr(arguments, 'score_inputs', [])
        ]

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(fitted_pipeline.problem_description)

    if getattr(arguments, 'scoring_params', None) is not None:
        scoring_params = {name: value for name, value in arguments.scoring_params}
    else:
        scoring_params = {}

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    predictions, produce_result = produce(fitted_pipeline, test_inputs, expose_produced_outputs=expose_produced_outputs)

    if produce_result.has_error():
        _output_pipeline_runs(arguments, [produce_result.pipeline_run])
        produce_result.check_success()
        assert False

    if getattr(arguments, 'output', None) is not None:
        predictions.to_csv(arguments.output)

    if expose_produced_outputs:
        save_steps_outputs(produce_result, arguments.expose_produced_outputs_dir)

    scores, score_result = score(
        scoring_pipeline,
        fitted_pipeline.problem_description,
        predictions,
        score_inputs,
        metrics,
        fitted_pipeline.random_seed,
        scoring_params=scoring_params,
        context=context,
        random_seed=getattr(arguments, 'random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run, score_inputs=score_inputs,
    )

    if score_result.has_error():
        _output_pipeline_runs(arguments, [produce_result.pipeline_run])
        score_result.check_success()
        assert False

    # Modifies "produce_pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, metrics=metrics, scores=scores,
    )

    _output_pipeline_runs(arguments, [produce_result.pipeline_run])

    if getattr(arguments, 'scores', None) is not None:
        scores.to_csv(arguments.scores)


def fit_produce_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    pipeline = pipeline_resolver(
        arguments.pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None),
            compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['train_inputs']
        test_inputs = meta['test_inputs']
    else:
        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem)
        else:
            problem_description = None

        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]
        test_inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'test_inputs', [])
        ]

    is_standard_pipeline = getattr(arguments, 'standard_pipeline', True)

    fitted_pipeline, predictions, fit_result = fit(
        pipeline, problem_description, inputs, context=context,
        random_seed=getattr(arguments, 'random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
        is_standard_pipeline=is_standard_pipeline,
    )

    if fit_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run])
        fit_result.check_success()
        assert False

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    predictions, produce_result = produce(fitted_pipeline, test_inputs, expose_produced_outputs=expose_produced_outputs)

    _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])

    produce_result.check_success()

    if getattr(arguments, 'output', None) is not None:
        assert is_standard_pipeline
        predictions.to_csv(arguments.output)

    if expose_produced_outputs:
        save_steps_outputs(produce_result, arguments.expose_produced_outputs_dir)


def fit_score_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    pipeline = pipeline_resolver(
        arguments.pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )
    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['train_inputs']
        test_inputs = meta['test_inputs']
        score_inputs = meta['score_inputs']
    else:
        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem)
        else:
            problem_description = None

        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]
        test_inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'test_inputs', [])
        ]
        score_inputs = [
            dataset_resolver(
                score_input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for score_input_uri in getattr(arguments, 'score_inputs', [])
        ]

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(problem_description)

    if getattr(arguments, 'scoring_params', None) is not None:
        scoring_params = {name: value for name, value in arguments.scoring_params}
    else:
        scoring_params = {}

    fitted_pipeline, predictions, fit_result = fit(
        pipeline, problem_description, inputs, context=context,
        random_seed=getattr(arguments, 'random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    if fit_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run])
        fit_result.check_success()
        assert False

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    predictions, produce_result = produce(fitted_pipeline, test_inputs, expose_produced_outputs=expose_produced_outputs)

    if produce_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])
        produce_result.check_success()
        assert False

    if getattr(arguments, 'output', None) is not None:
        predictions.to_csv(arguments.output)

    if expose_produced_outputs:
        save_steps_outputs(produce_result, arguments.expose_produced_outputs_dir)

    scores, score_result = score(
        scoring_pipeline, problem_description, predictions, score_inputs, metrics, fitted_pipeline.random_seed,
        scoring_params=scoring_params, context=context,
        random_seed=getattr(arguments, 'scoring_random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run, score_inputs=score_inputs,
    )

    if score_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])
        score_result.check_success()
        assert False

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, metrics=metrics, scores=scores,
    )

    _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])

    if getattr(arguments, 'scores', None) is not None:
        scores.to_csv(arguments.scores)


def score_predictions_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        score_inputs = meta['score_inputs']
    else:
        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem)
        else:
            problem_description = None

        score_inputs = [
            dataset_resolver(
                score_input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for score_input_uri in getattr(arguments, 'score_inputs', [])
        ]

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(problem_description)

    if getattr(arguments, 'scoring_params', None) is not None:
        scoring_params = {name: value for name, value in arguments.scoring_params}
    else:
        scoring_params = {}

    predictions_dataframe = pandas.read_csv(
        arguments.predictions,
        # We do not want to do any conversion of values at this point.
        # This should be done by primitives later on.
        dtype=str,
        # We always expect one row header.
        header=0,
        # We want empty strings and not NaNs.
        na_filter=False,
        encoding='utf8',
        low_memory=False,
        memory_map=True,
    )

    # Convert pandas DataFrame to container DataFrame.
    predictions = container.DataFrame(predictions_dataframe, generate_metadata=True)

    if getattr(arguments, 'output', None) is not None:
        predictions.to_csv(arguments.output)

    scores, score_result = score(
        scoring_pipeline, problem_description, predictions, score_inputs, metrics,
        getattr(arguments, 'predictions_random_seed', None),
        scoring_params=scoring_params,
        context=context,
        random_seed=getattr(arguments, 'scoring_random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    score_result.check_success()

    if getattr(arguments, 'scores', None) is not None:
        scores.to_csv(arguments.scores)


def evaluate_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    pipeline = pipeline_resolver(
        arguments.pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )
    data_pipeline = pipeline_resolver(
        arguments.data_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )
    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['full_inputs']
    else:
        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem)
        else:
            problem_description = None

        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]

    if getattr(arguments, 'data_params', None) is not None:
        data_params = {name: value for name, value in arguments.data_params}
    else:
        data_params = {}

    if getattr(arguments, 'data_split_file', None) is not None:
        split_file = pandas.read_csv(
            arguments.data_split_file,
            # We do not want to do any conversion of values at this point.
            # This should be done by primitives later on.
            dtype=str,
            # We always expect one row header.
            header=0,
            # We want empty strings and not NaNs.
            na_filter=False,
            encoding='utf8',
            low_memory=False,
            memory_map=True,
        )

        # We use just the "d3mIndex" column and ignore multi-key indices.
        # This works for now because it seems that every current multi-key
        # dataset in fact has an unique value in "d3mIndex" alone.
        # See: https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/issues/117
        # Hyper-parameter value has to be JSON-serialized.
        data_params['primary_index_values'] = json.dumps(list(split_file.loc[split_file['type'] == 'TEST']['d3mIndex']))

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(problem_description)

    if getattr(arguments, 'scoring_params', None) is not None:
        scoring_params = {name: value for name, value in arguments.scoring_params}
    else:
        scoring_params = {}

    scores_list, results_list = evaluate(
        pipeline, data_pipeline, scoring_pipeline, problem_description, inputs, data_params, metrics,
        scoring_params=scoring_params,
        context=context, random_seed=getattr(arguments, 'random_seed', 0),
        data_random_seed=getattr(arguments, 'data_random_seed', 0),
        scoring_random_seed=getattr(arguments, 'scoring_random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    _output_pipeline_runs(arguments, results_list.pipeline_runs)

    results_list.check_success()

    scores = combine_folds(scores_list)

    if getattr(arguments, 'scores', None) is not None:
        scores.to_csv(arguments.scores)


def save_steps_outputs(results: typing.Union[Result, MultiResult], output_dir: str) -> None:
    if isinstance(results, Result):
        for key, step_output in results.values.items():
            container_utils.save_container(step_output, os.path.join(output_dir, key))
    elif isinstance(results, MultiResult):
        for i, result in enumerate(results):
            for key, step_output in result.values.items():
                container_utils.save_container(step_output, os.path.join(output_dir, str(i), key))
    else:
        raise exceptions.UnexpectedTypeError("Type: {results_type}".format(results_type=type(results)))


def main(argv: typing.Sequence) -> None:
    # We have to disable importing while type checking because it makes
    # an import cycle in mypy which makes many typing errors.
    if not typing.TYPE_CHECKING:
        # Importing here to prevent import cycle.
        # from d3m import cli
        from dsbox.template import cli

        logging.basicConfig()

        logger.warning("This CLI is deprecated. Use \"python3 -m d3m runtime\" instead.")

        parser = argparse.ArgumentParser(description="Run D3M pipelines.")
        cli.runtime_configure_parser(parser)

        arguments = parser.parse_args(argv[1:])
        cli.runtime_handler(arguments, parser)


if __name__ == '__main__':
    if 'DSBOX_LOGGING_LEVEL' not in os.environ or 'DSBOX_LOGGING_DIR' not in os.environ:
        print('To use DSBox logging do:')
        print('  export DSBOX_LOGGING_LEVEL="dsbox=WARNING:dsbox.template.runtime=DEBUG:console_logging_level=WARNING:file_logging_level=DEBUG"')
        print('  export DSBOX_LOGGING_DIR=$HOME/output')
        sys.exit(1)
    else:
        # set logging level
        from dsbox.controller.config import DsboxConfig
        DsboxConfig()._load_logging()

        os.makedirs(os.environ['DSBOX_LOGGING_DIR'] + '/dfs', exist_ok=True)
    main(sys.argv)
