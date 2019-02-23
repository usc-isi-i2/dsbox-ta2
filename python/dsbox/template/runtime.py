import os
import typing
import contextlib
import logging
import sys
import tempfile
import time
import pdb

from pandas import DataFrame  # type: ignore
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore

import d3m.runtime as runtime_base
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, pipeline_run as pipeline_run_module, problem

from d3m.primitive_interfaces import base
from d3m import exceptions
from multiprocessing import current_process
from dsbox.JobManager.cache import PrimitivesCache

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
            self, pipeline: pipeline_module.Pipeline,  hyperparams: typing.Sequence = None,
            *,
            problem_description: typing.Dict = None, context: metadata_base.Context = metadata_base.Context.TESTING,
            random_seed: int = 0, volumes_dir: str = None, is_standard_pipeline: bool = False,
            environment: pipeline_run_module.RuntimeEnvironment = None,
            users: typing.Sequence[pipeline_run_module.User] = None,
            fitted_pipeline_id: str = None, template_name: str = '', log_dir: str = None
    ) -> None:

        super().__init__(
            pipeline=pipeline, hyperparams=hyperparams, problem_description=problem_description, context=context,
            random_seed=random_seed, volumes_dir=volumes_dir, is_standard_pipeline=is_standard_pipeline,
            environment=environment, users=users)
        # def __init__(self, pipeline_description: Pipeline, fitted_pipeline_id: str, log_dir) -> None:

        # super().__init__(pipeline=pipeline_description, hyperparams=None, problem_description=None)

        self.cache: PrimitivesCache = None
        self.cross_validation_result = None
        self.fitted_pipeline_id = fitted_pipeline_id
        self.template_name = template_name
        self.fit_outputs = None
        self.log_dir = log_dir
        self.metric_descriptions = None
        self.produce_outputs = None
        self.timing = {}
        self.timing["total_time_used"] = 0.0

        self.use_cache = True
        # self.timing["total_time_used_without_cache"] = 0.0

        # !
        self.skip_fit_phase = False

    def set_not_use_cache(self) -> None:
        self.use_cache = False

    def set_metric_descriptions(self, metric_descriptions):
        self.metric_descriptions = metric_descriptions

    def _run_primitive(self, this_step: pipeline_module.PrimitiveStep) -> None:
        '''
            Override the d3m_runtime's function
            And add the cache support
        '''
        if this_step.primitive is None:
            raise exceptions.InvalidPipelineError("Primitive has not been resolved.")

        time_start = time.time()

        _logger.debug(f"running primitive: {this_step.primitive.metadata.query()['name']}")
        # call d3m's run primitive directly if not use cache
        # NOTE: But need to perform cross validation!
        if not self.use_cache:
            super()._run_primitive(this_step)

        elif self.phase == metadata_base.PipelineRunPhase.FIT:

            # Same as old codes, use argument as the cache system's key
            primitive_arguments = self._prepare_primitive_arguments(this_step)
            primitive_arguments["produce_methods"] = this_step.outputs
            prim_name, prim_hash = self.cache._get_hash(
                hash_prefix=None, pipe_step=self.pipeline.steps[self.current_step],
                primitive_arguments=primitive_arguments)

            if not self.skip_fit_phase:
                # if we need to do cross validation, do it before normal fit() step
                if '_dsbox_runtime' in this_step.__dict__ and "cross_validation" in this_step._dsbox_runtime:

                    primitive: typing.Type[base.PrimitiveBase] = this_step.primitive
                    # TODO: add one more "if" to restrict runtime to run cross validation only for tuning steps
                    primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    custom_hyperparams = dict()
                    # produce_params only have 'inputs'
                    produce_params = dict((k, primitive_arguments[k]) for k in ["inputs"])
                    # training_arguments have ['inputs', 'outputs']
                    training_arguments = dict((k, primitive_arguments[k]) for k in ["inputs","outputs"])

                    if bool(this_step.hyperparams):
                        for hyperparam, value in this_step.hyperparams.items():
                            if isinstance(value, dict):
                                custom_hyperparams[hyperparam] = value['data']
                            else:
                                custom_hyperparams[hyperparam] = value

                    # self.cross_validation_result = None
                    self.cross_validation_result = self._cross_validation(
                        primitive, training_arguments, produce_params, primitive_hyperparams,
                        custom_hyperparams, this_step._dsbox_runtime)

                    # print("!@#$$%$$$,cvfinished!!!")
                    # print(self.cross_validation_result)
                # END for cross-validation process

            cache_hit = False
            _logger.debug(
                "Primitive Fit. 'id': '%(primitive_id)s', '(name, hash)': ('%(name)s', "
                "'%(hash)s'), 'worker_id': '%(worker_id)s'.",
                {
                    'primitive_id':
                        self.pipeline.steps[self.current_step].get_primitive_id(),
                    'name': prim_name,
                    'hash': prim_hash,
                    'worker_id': current_process()
                },
            )

            # if this primitive hitted
            if self.cache.is_hit_key(prim_hash=prim_hash, prim_name=prim_name):
                # TODO: How to handle pipeline_run for cache hit?
                self.pipeline_run.add_primitive_step(this_step)

                fitting_time, model = self.cache.lookup_key(prim_name=prim_name, prim_hash=prim_hash)
                self.steps_state[self.current_step] = model

                # print cache reading time
                cache_reading_time = (time.time() - time_start)
                _logger.debug(f"[INFO] cache reading took {cache_reading_time} s and "
                              f"fitting time took {fitting_time} s")
                cache_hit = True
                # print("!!!!Fit step with hitted finished!!!!")

                # HERE the code adapted from d3m's runtime!!! If new version of runtime changd,
                # remember to change here
                # this part use the primitive adapted from cache to regenerate the prediction of
                # training dataset
                # and output the results to self.environment which is used to store the
                # intermediate results of each steps

                # === From v2018.7
                # multi_produce_arguments = self._filter_arguments(this_step.primitive,
                #                                                  'multi_produce',
                #                                                  dict(primitive_arguments,
                #                                                       produce_methods=this_step.outputs))
                # while True:
                #     multi_call_result = model.multi_produce(**multi_produce_arguments)
                #     if multi_call_result.has_finished:
                #         outputs = multi_call_result.values
                #         break
                # for output_id in this_step.outputs:
                #     output_data_reference = 'steps.{i}.{output_id}'.format(i=this_step.index,
                #                                                            output_id=output_id)
                #     self.environment[output_data_reference] = outputs[output_id]

                primitive = model
                # === From v2019.1.21
                fit_multi_produce_arguments = self._filter_arguments(this_step.primitive, 'fit_multi_produce', dict(primitive_arguments, produce_methods=this_step.outputs))
                while True:
                    multi_call_result = self._call_primitive_method(primitive.fit_multi_produce, fit_multi_produce_arguments)
                    if multi_call_result.has_finished:
                        outputs = multi_call_result.values
                        break

                for output_id in this_step.outputs:
                    output_data_reference = 'steps.{i}.{output_id}'.format(i=this_step.index, output_id=output_id)

                    if output_id in outputs:
                        self.data_values[output_data_reference] = outputs[output_id]
                    else:
                        raise exceptions.InvalidReturnValueError("Missing declared output '{output_id}' in computed primitive's outputs.".format(output_id=output_id))


                # if we did not find the cache, run the primitive with d3m's inner function
            else:
                # print("!!!!Fit step with not hit!!!!")
                super()._run_primitive(this_step)
                fitting_time = (time.time() - time_start)
                # get the model after fitting
                model = self.steps_state[self.current_step]
                # push the model to cache
                self.cache.push_key(prim_name=prim_name, prim_hash=prim_hash, model=model,
                                    fitting_time=fitting_time)
                # log fitting results
                for output_id in this_step.outputs:
                    output_data_reference = 'steps.{i}.{output_id}'.format(i=this_step.index, output_id=output_id)
                self._log_fitted_step(self.cache, output_data_reference, this_step, self.data_values)
            # END processing part for FIT Phase

        elif self.phase == metadata_base.PipelineRunPhase.PRODUCE:
            # if in produce step, always use the d3m's codes
            super()._run_primitive(this_step)

            for output_id in this_step.outputs:
                output_data_reference = 'steps.{i}.{output_id}'.format(i=this_step.index, output_id=output_id)
            self._log_produce_step(output_data_reference, this_step, self.data_values)

        else:
            raise exceptions.UnexpectedValueError("Unknown phase: {phase}".format(phase=self.phase))

        # add up the timing
        self.timing["total_time_used"] += (time.time() - time_start)
        _logger.debug(f"   done primitive: {this_step.primitive.metadata.query()['name']}")

    def _log_fitted_step(self, cache, output_data_reference, primitive_step, primitives_outputs):
        '''
            The function use to record the intermediate output of each primitive and save into
            the logs

            Parameters
            ---------
            cache: indicate the cache object
            output_data_reference: a str use to indicate the dict key of the step output stored
            in self.environment
            primitive_step: indicate the primitive for logging
            primitives_outputs: the dict used to store outputs
        '''
        if _logger.getEffectiveLevel() <= 10:

            # _logger.debug('cache keys')
            # for key in sorted(cache.storage.keys()):
            #     _logger.debug('   {}'.format(key))

            n_step = int(output_data_reference.split('.')[1])
            debug_file = os.path.join(
                self.log_dir, 'dfs',
                'fit_{}_{}_{}_{:02}_{}'.format(self.template_name, self.pipeline.id.split('-')[0],
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
                        metadata_filepath = debug_file + '_meta'
                        with open(metadata_filepath, 'w') as out:
                            primitives_outputs[output_data_reference].metadata.pretty_print(handle=out)
                    except Exception:
                        pass

    def _log_produce_step(self, output_data_reference, primitive_step, primitives_outputs):
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
                'pro_{}_{}_{}_{:02}_{}'.format(self.template_name, self.pipeline.id.split('-')[0],
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
                        metadata_filepath = debug_file + '_meta'
                        with open(metadata_filepath, 'w') as out:
                            primitives_outputs[output_data_reference].metadata.pretty_print(handle=out)
                    except Exception:
                        pass

    def _cross_validation(self, primitive: typing.Type[base.PrimitiveBase],
                          training_arguments: typing.Dict,
                          produce_params: typing.Dict,
                          primitive_hyperparams: hyperparams_module.Hyperparams,
                          custom_hyperparams: typing.Dict,
                          runtime_instr: typing.Dict,
                          seed: int = 4767) -> typing.List:

        _logger.debug('cross-val primitive: %s' % str(primitive))

        results: typing.List[str, typing.Dict] = []

        validation_metrics: typing.Dict[str, typing.List[float]] = defaultdict(list)
        targets: typing.Dict[str, typing.List[list]] = defaultdict(list)

        X = training_arguments['inputs']
        y = training_arguments['outputs']

        cv = runtime_instr.get('cross_validation', 10)
        use_stratified = runtime_instr.get('stratified', False)

        # TODO: cross validation need to be update to fit with new requirement with adding indexes!!

        # Redirect stderr to an error file
        #  Directly assigning stderr to tempfile.TemporaryFile cause printing str to fail
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, str(primitive)), 'w') as errorfile:
                with contextlib.redirect_stderr(errorfile):

                    if use_stratified:
                        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
                    else:
                        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

                    num = 0.0
                    for k, (train, test) in enumerate(kf.split(X, y)):
                        try:
                            # !!!
                            # Temporary fix
                            # Still ignore the use_semantic types hyperparameters
                            if "use_semantic_types" in custom_hyperparams:
                                custom_hyperparams.pop("use_semantic_types")
                            if "return_result" in custom_hyperparams:
                                custom_hyperparams.pop("return_result")
                            if "add_index_columns" in custom_hyperparams:
                                custom_hyperparams.pop("add_index_columns")

                            model = primitive(hyperparams=primitive_hyperparams(
                                primitive_hyperparams.defaults(), **custom_hyperparams))
                        except Exception:
                            print(
                                "******************\n[ERROR]Hyperparameters unsuccesfully set - "
                                "using defaults")
                            model = primitive(
                                hyperparams=primitive_hyperparams(primitive_hyperparams.defaults()))

                        if model is None:
                            return results

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

                        validation_train = dict(training_arguments)
                        validation_train['inputs'] = trainX
                        validation_train['outputs'] = trainY

                        validation_test = dict(produce_params)
                        validation_test['inputs'] = testX

                        try:
                            model.set_training_data(**validation_train)
                            model.fit()
                            ypred = model.produce(**validation_test).value

                            num = num + 1.0

                            targets['ground_truth'].append(testY)
                            targets['prediction'].append(ypred)
                            for metric_description in self.metric_descriptions:
                                metricDesc = problem.PerformanceMetric.parse(metric_description['metric'])
                                metric: typing.Callable = metricDesc.get_function()
                                params: typing.Dict = metric_description['params']
                                validation_metrics[metric_description['metric']].append(
                                    metric(testY, ypred, **params))

                        except Exception as e:
                            sys.stderr.write(
                                "ERROR: cross_validation {}: {}\n".format(primitive, e))
                            # traceback.print_exc(e)

        if num == 0:
            return results

        average_metrics: typing.Dict[str, dict] = {}
        for name, values in validation_metrics.items():
            if len(values) == 0:
                return results
            average_metrics[name] = sum(values) / len(values)

        for metric_description in self.metric_descriptions:
            result_by_metric = {}
            result_by_metric['metric'] = metric_description['metric']
            result_by_metric['value'] = average_metrics[metric_description['metric']]
            result_by_metric['values'] = validation_metrics[metric_description['metric']]
            result_by_metric['targets'] = targets[metric_description['metric']]
            results.append(result_by_metric)

        for result in results:
            _logger.debug('cross-validation metric: %s=%.4f', result['metric'], result['value'])
            _logger.debug('cross-validation details: %s %s',
                          result['metric'], str(['%.4f' % x for x in result['values']]))

        return results

    def fit(self, **arguments: typing.Any) -> None:
        """
        Train all steps in the pipeline.

        Paramters
        ---------
        arguments
            Arguments required to train the Pipeline
        """
        if 'cache' in arguments:
            _logger.debug("Using global cache")
            self.cache: PrimitivesCache = arguments['cache']
        else:
            _logger.debug("Using local cache")
            self.cache = PrimitivesCache()

        self.fit_outputs = super().fit(inputs=arguments['inputs'])
        self.check_results(self.fit_outputs)
        return self.fit_outputs

    def produce(self, **arguments: typing.Any) -> typing.List:
        """
        Train all steps in the pipeline.

        Paramters
        ---------
        arguments
            Arguments required to execute the Pipeline
        """

        self.produce_outputs = super().produce(inputs=arguments['inputs'])
        self.check_results(self.produce_outputs)
        return self.produce_outputs

    def check_results(self, res: runtime_base.Result):
        if res.error is None:
            assert len(res.values) > 0
        else:
            raise res.error
