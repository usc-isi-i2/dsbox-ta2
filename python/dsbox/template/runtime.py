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
        if fitted_pipeline_id is None:
            # Occurs when runtime is not initialized by DSBox
            self.fitted_pipeline_id = pipeline.id
        else:
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

                self._check_primitive_output(this_step, self.data_values)

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

    def _check_primitive_output(self, primitive_step, primitives_outputs):
        for output_id in primitive_step.outputs:
            output_data_reference = 'steps.{i}.{output_id}'.format(i=primitive_step.index, output_id=output_id)
            output = primitives_outputs[output_data_reference]
            if isinstance(output, DataFrame):
                row_size, col_size = primitives_outputs[output_data_reference].shape
                for col in range(col_size):
                    if len(output.metadata.query((metadata_base.ALL_ELEMENTS, col))) == 0:
                        _logger.warning(f'Incomplete metadata at col {col}. Primitive={primitive_step.primitive}')

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
                            _logger.error("ERROR: cross_validation {}: {}\n".format(primitive, e))
                            # traceback.print_exc(e)

        if num == 0:
            return results

        average_metrics: typing.Dict[str, dict] = {}
        for name, values in validation_metrics.items():
            if len(values) == 0:
                return results
            average_metrics[name] = sum(values) / len(values)

        for metric_description in self.validation_metrics:
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
            self.cache: PrimitivesCache = arguments['cache']
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

import argparse
import json
import frozendict  # type: ignore
import pandas
import pickle
import uuid
from urllib import parse as url_parse
from pathlib import Path

from d3m import container, utils
from d3m.container import dataset as dataset_module

logger = logging.getLogger(__name__)

def _prepare_hyperparams(free_hyperparams: typing.Sequence, hyperparameter_values: typing.Dict) -> typing.Tuple[typing.Sequence, typing.Set[str]]:
    """
    Values in ``hyperparameter_values`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json`` method call.
    """

    hyperparams: typing.List[typing.Union[typing.Dict, typing.Sequence]] = []

    hyperparameter_values_used = set()

    for free_hyperparams_for_step in free_hyperparams:
        if isinstance(free_hyperparams_for_step, (dict, frozendict.frozendict)):
            values = {}
            for name, hyperparameter in free_hyperparams_for_step.items():
                if name in hyperparameter_values:
                    values[name] = hyperparameter.value_from_json(json.loads(hyperparameter_values[name]))
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
    pipeline: pipeline_module.Pipeline, problem_description: typing.Dict, inputs: typing.Sequence[container.Dataset], *,
    context: metadata_base.Context, hyperparams: typing.Sequence = None, random_seed: int = 0, volumes_dir: str = None,
    runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
    log_dir=None
) -> typing.Tuple[Runtime, container.DataFrame, pipeline_run_module.PipelineRun]:
    for input in inputs:
        if not isinstance(input, container.Dataset):
            raise TypeError("A standard pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(input),
            ))

    if len(pipeline.outputs) != 1:
        raise ValueError("A standard pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(pipeline.outputs),
        ))

    runtime = Runtime(
        pipeline, hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir,
        is_standard_pipeline=True, environment=runtime_environment,
        log_dir=log_dir
    )

    result = runtime.fit(inputs, return_values=['outputs.0'])
    result.check_success()

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A standard pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    return runtime, output, result.pipeline_run


# TODO: Add debug logging.
def produce(
    fitted_pipeline: Runtime, test_inputs: typing.Sequence[container.Dataset],
) -> typing.Tuple[container.DataFrame, pipeline_run_module.PipelineRun]:
    for test_input in test_inputs:
        if not isinstance(test_input, container.Dataset):
            raise TypeError("A standard pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(test_input),
            ))

    # This is checked in "fit" already, but maybe somebody fitter a pipeline not through "fit".
    if len(fitted_pipeline.pipeline.outputs) != 1:
        raise ValueError("A standard pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(fitted_pipeline.pipeline.outputs),
        ))

    result = fitted_pipeline.produce(test_inputs, return_values=['outputs.0'])
    result.check_success()

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A standard pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    return output, result.pipeline_run


# TODO: Add debug logging.
def score(
    scoring_pipeline: pipeline_module.Pipeline, problem_description: typing.Dict, predictions: container.DataFrame,
    score_inputs: typing.Sequence[container.Dataset], metrics: typing.Sequence[typing.Dict], *,
    context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[container.DataFrame, pipeline_run_module.PipelineRun]:
    for score_input in score_inputs:
        if not isinstance(score_input, container.Dataset):
            raise TypeError("A scoring pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(score_input),
            ))

    if len(scoring_pipeline.outputs) != 1:
        raise ValueError("A scoring pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(scoring_pipeline.outputs),
        ))

    if not metrics:
        raise exceptions.InvalidArgumentValueError("A list of metrics for scores to compute cannot be empty.")

    metrics_hyperparameter = []
    for metric in metrics:
        metric_hyperparameter = {'metric': metric['metric'].name, 'k': None, 'pos_label': None}
        metric_hyperparameter.update(metric.get('params', {}))
        metrics_hyperparameter.append(metric_hyperparameter)

    scoring_params = {
        # We have to JSON-serialize it because "_prepare_hyperparams" expects
        # all values to be JSON-serialized.
        'metrics': json.dumps(metrics_hyperparameter),
    }

    hyperparams, scoring_params_used = _prepare_hyperparams(scoring_pipeline.get_free_hyperparams(), scoring_params)

    scoring_params_keys_set = set(scoring_params.keys())
    if scoring_params_keys_set - scoring_params_used:
        logger.warning("Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s", {
            'pipeline_id': scoring_pipeline.id,
            'unused_params': sorted(scoring_params_keys_set - scoring_params_used),
        })

    runtime = Runtime(
        scoring_pipeline, hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir, environment=runtime_environment,
    )

    inputs = [predictions] + list(score_inputs)  # type: ignore

    # Fit + produce on same data.
    result = runtime.fit(inputs, return_values=['outputs.0'])
    result.check_success()

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A scoring pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    return output, result.pipeline_run


# TODO: Add debug logging.
def prepare_data(
    data_pipeline: pipeline_module.Pipeline, problem_description: typing.Dict, inputs: typing.Sequence[container.Dataset],
    data_params: typing.Dict[str, str], *, context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.List, pipeline_run_module.PipelineRun]:
    """
    Values in ``data_params`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json`` method call.
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
        environment=runtime_environment,
    )

    # Fit + produce on same data. The inputs are the list of indices of folds
    # to generate and a dataset to split.
    result = runtime.fit([container.List(range(number_of_folds))] + list(inputs), return_values=['outputs.0', 'outputs.1', 'outputs.2'])  # type: ignore
    result.check_success()

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

    return outputs, result.pipeline_run


# TODO: Add debug logging.
def evaluate(
    pipeline: pipeline_module.Pipeline, data_pipeline: pipeline_module.Pipeline,
    scoring_pipeline: pipeline_module.Pipeline, problem_description: typing.Dict,
    inputs: typing.Sequence[container.Dataset], data_params: typing.Dict[str, str],
    metrics: typing.Sequence[typing.Dict], *, context: metadata_base.Context,
    hyperparams: typing.Sequence = None, random_seed: int = 0, data_random_seed: int = 0,
    scoring_random_seed: int = 0, volumes_dir: str = None,
    runtime_environment: pipeline_run_module.RuntimeEnvironment = None
) -> typing.List[typing.Tuple[container.DataFrame, pipeline_run_module.PipelineRun, pipeline_run_module.PipelineRun]]:
    """
    Values in ``data_params`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json`` method call.
    """

    outputs, data_pipeline_run = prepare_data(
        data_pipeline, problem_description, inputs, data_params,
        context=context, random_seed=data_random_seed, volumes_dir=volumes_dir,
        runtime_environment=runtime_environment,
    )
    fold_group_uuid = uuid.uuid4()

    results_list = []
    all_pipeline_runs: typing.List[pipeline_run_module.PipelineRun] = []
    for fold_index, (train_inputs, test_inputs, score_inputs) in enumerate(zip(*outputs)):
        try:
            fitted_pipeline, predictions, fit_pipeline_run = fit(
                pipeline, problem_description, [train_inputs], context=context, hyperparams=hyperparams,
                random_seed=random_seed, volumes_dir=volumes_dir, runtime_environment=runtime_environment,
            )
        except exceptions.PipelineRunError as error:
            if error.pipeline_runs:
                assert len(error.pipeline_runs) == 1, len(error.pipeline_runs)

                # Modifies "error.pipeline_runs[0]" in-place.
                combine_pipeline_runs(
                    error.pipeline_runs[0], data_pipeline_run=data_pipeline_run,
                    fold_group_uuid=fold_group_uuid, fold_index=fold_index
                )

            error.pipeline_runs = all_pipeline_runs + list(error.pipeline_runs)

            raise error

        # Modifies "fit_pipeline_run" in-place.
        combine_pipeline_runs(
            fit_pipeline_run, data_pipeline_run=data_pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index
        )

        all_pipeline_runs.append(fit_pipeline_run)

        try:
            predictions, produce_pipeline_run = produce(fitted_pipeline, [test_inputs])
        except exceptions.PipelineRunError as error:
            if error.pipeline_runs:
                assert len(error.pipeline_runs) == 1, len(error.pipeline_runs)

                # Modifies "error.pipeline_runs[0]" in-place.
                combine_pipeline_runs(
                    error.pipeline_runs[0], data_pipeline_run=data_pipeline_run,
                    fold_group_uuid=fold_group_uuid, fold_index=fold_index
                )

            error.pipeline_runs = all_pipeline_runs + list(error.pipeline_runs)

            raise error

        # Modifies "produce_pipeline_run" in-place.
        combine_pipeline_runs(
            produce_pipeline_run, data_pipeline_run=data_pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index,
        )

        all_pipeline_runs.append(produce_pipeline_run)

        try:
            scores, scoring_pipeline_run = score(
                scoring_pipeline, problem_description, predictions, [score_inputs], metrics,
                context=context, random_seed=scoring_random_seed, volumes_dir=volumes_dir,
                runtime_environment=runtime_environment,
            )
        except exceptions.PipelineRunError as error:
            if error.pipeline_runs:
                assert len(error.pipeline_runs) == 1, len(error.pipeline_runs)

                # Modifies "produce_pipeline_run" in-place.
                combine_pipeline_runs(
                    produce_pipeline_run, scoring_pipeline_run=error.pipeline_runs[0],
                )

            # We modified "produce_pipeline_run" in-place and "produce_pipeline_run"
            # is already among "all_pipeline_runs", so we can just set it.
            error.pipeline_runs = all_pipeline_runs

            raise error

        # Modifies "produce_pipeline_run" in-place.
        combine_pipeline_runs(
            produce_pipeline_run, scoring_pipeline_run=scoring_pipeline_run,
            metrics=metrics, scores=scores, problem_description=problem_description,
        )

        results_list.append((scores, fit_pipeline_run, produce_pipeline_run))

    return results_list


def get_pipeline(
    pipeline_path: str, *, strict_resolving: bool = False, strict_digest: bool = False,
    pipeline_search_paths: typing.Sequence[str] = None, respect_environment_variable: bool = True, load_all_primitives: bool = True,
    resolver_class: typing.Type[pipeline_module.Resolver] = pipeline_module.Resolver,
    pipeline_class: typing.Type[pipeline_module.Pipeline] = pipeline_module.Pipeline,
) -> pipeline_module.Pipeline:
    resolver = resolver_class(
        strict_resolving=strict_resolving, strict_digest=strict_digest, pipeline_search_paths=pipeline_search_paths,
        respect_environment_variable=respect_environment_variable, load_all_primitives=load_all_primitives,
    )

    if os.path.exists(pipeline_path):
        with open(pipeline_path, 'r', encoding='utf8') as pipeline_file:
            if pipeline_path.endswith('.yml'):
                return pipeline_class.from_yaml(pipeline_file, resolver=resolver, strict_digest=strict_digest)
            elif pipeline_path.endswith('.json'):
                return pipeline_class.from_json(pipeline_file, resolver=resolver, strict_digest=strict_digest)
            else:
                raise ValueError("Unknown file extension.")
    else:
        return resolver.get_pipeline({'id': pipeline_path})


def is_uri(uri: str) -> bool:
    """
    Test if a given string is an URI.

    Parameters
    ----------
    uri : str
        A potential URI to test.

    Returns
    -------
    bool
        ``True`` if string is an URI, ``False`` otherwise.
    """

    try:
        parsed_uri = url_parse.urlparse(uri)
    except Exception:
        return False

    return parsed_uri.scheme != ''


def get_dataset(dataset_uri: str, *, compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False) -> container.Dataset:
    if not is_uri(dataset_uri):
        dataset_uri = 'file://{dataset_doc_path}'.format(dataset_doc_path=os.path.abspath(dataset_uri))

    return container.Dataset.load(dataset_uri, compute_digest=compute_digest, strict_digest=strict_digest)


# TODO: Do not traverse the datasets directory every time.
def parse_meta(meta_file: typing.TextIO, datasets_dir: str, *, dataset_resolver: typing.Callable = None,
               compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False,
               merge_score_targets: bool = True) -> typing.Dict:
    if dataset_resolver is None:
        dataset_resolver = get_dataset

    if datasets_dir is None:
        raise exceptions.InvalidArgumentValueError("Dataset directory has to be provided to resolve meta files.")

    meta = json.load(meta_file)

    datasets: typing.Dict[str, str] = {}
    problem_descriptions: typing.Dict[str, str] = {}

    for dirpath, dirnames, filenames in os.walk(datasets_dir, followlinks=True):
        dirpath = os.path.abspath(os.path.join(datasets_dir, dirpath))

        if 'datasetDoc.json' in filenames:
            # Do not traverse further (to not parse "datasetDoc.json" or "problemDoc.json" if they
            # exists in raw data filename).
            dirnames[:] = []

            dataset_path = os.path.join(dirpath, 'datasetDoc.json')

            try:
                with open(dataset_path, 'r', encoding='utf8') as dataset_file:
                    dataset_doc = json.load(dataset_file)

                dataset_id = dataset_doc['about']['datasetID']
                # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
                # They are the same as TEST dataset splits, but we present them differently, so that
                # SCORE dataset splits have targets as part of data. Because of this we also update
                # corresponding dataset ID.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
                if merge_score_targets and os.path.exists(os.path.join(dirpath, '..', 'targets.csv')) and dataset_id.endswith('_TEST'):
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

            problem_path = os.path.join(dirpath, 'problemDoc.json')

            try:
                with open(problem_path, 'r', encoding='utf8') as problem_file:
                    problem_doc = json.load(problem_file)

                problem_id = problem_doc['about']['problemID']
                # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
                # They are the same as TEST dataset splits, but we present them differently, so that
                # SCORE dataset splits have targets as part of data. Because of this we also update
                # corresponding problem ID.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
                if merge_score_targets and os.path.exists(os.path.join(dirpath, '..', 'targets.csv')) and problem_id.endswith('_TEST'):
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
        'problem': problem.parse_problem_description(problem_descriptions[meta['problem']]),
        'full_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['full_inputs']],
        'train_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['train_inputs']],
        'test_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['test_inputs']],
        'score_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['score_inputs']],
    }


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
        }, for_value=scores_list[fold])
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
    }, for_value=scores)

    return scores


def combine_pipeline_runs(
    standard_pipeline_run: pipeline_run_module.PipelineRun, *,
    data_pipeline_run: pipeline_run_module.PipelineRun = None, scoring_pipeline_run: pipeline_run_module.PipelineRun = None,
    metrics: typing.Sequence[typing.Dict] = None, scores: container.DataFrame = None,
    problem_description: typing.Dict = None,
    fold_group_uuid: uuid.UUID = None, fold_index: int = None,
) -> None:
    fold_args_provided = (item is None for item in (fold_group_uuid, fold_index))
    if any(fold_args_provided) and not all(fold_args_provided):
        raise exceptions.InvalidArgumentValueError("If any of 'fold_group_uuid' and 'fold_index' are provided, they must all be provided.")

    scores_args_provided = (item is None for item in (scores, metrics, problem_description))
    if any(scores_args_provided) and not all(scores_args_provided):
        raise exceptions.InvalidArgumentValueError("If any of 'scores', 'metrics', and 'problem_description' are provided, they must all be provided.")

    if data_pipeline_run is not None:
        standard_pipeline_run.set_data_preparation_pipeline_run(data_pipeline_run)

    if fold_group_uuid is not None:
        standard_pipeline_run.set_fold_group(fold_group_uuid, fold_index)

    if scoring_pipeline_run is not None:
        standard_pipeline_run.set_scoring_pipeline_run(scoring_pipeline_run)

    if scores is not None:
        standard_pipeline_run.set_scores(scores, metrics, problem_description)


def export_dataframe(dataframe: container.DataFrame, output_file: typing.TextIO = None) -> typing.Optional[str]:
    column_names = []
    for column_index in range(len(dataframe.columns)):
        # We use column name from the DataFrame is metadata does not have it. This allows a bit more compatibility.
        column_names.append(dataframe.metadata.query_column(column_index).get('name', dataframe.columns[column_index]))

    return dataframe.to_csv(output_file, header=column_names, index=False)


def get_metrics_from_list(metrics: typing.Sequence[str]) -> typing.Sequence[typing.Dict]:
    return [{'metric': problem.PerformanceMetric[metric]} for metric in metrics]


def get_metrics_from_problem_description(problem_description: typing.Dict) -> typing.Sequence[typing.Dict]:
    return problem_description['problem'].get('performance_metrics', [])


def _output_pipeline_runs(arguments: argparse.Namespace, pipeline_runs: typing.Sequence[pipeline_run_module.PipelineRun]) -> None:
    if not getattr(arguments, 'output_run', None):
        return

    first = True
    for pipeline_run in pipeline_runs:
        pipeline_run.to_yaml(arguments.output_run, appending=not first)
        first = False


def _fit(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset

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
        problem_description = problem.parse_problem_description(arguments.problem)
        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]

    try:
        fitted_pipeline, predictions, pipeline_run = fit(
            pipeline, problem_description, inputs,
            context=context, random_seed=getattr(arguments, 'random_seed', 0),
            volumes_dir=getattr(arguments, 'volumes_dir', None),
            runtime_environment=runtime_environment,
        )
    except exceptions.PipelineRunError as error:
        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)

    _output_pipeline_runs(arguments, [pipeline_run])


# We have "pipeline_resolver" as an argument (even if we are not using it
# in this function) so that the signature is the same for all handlers.
def _produce(arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None) -> None:
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset

    fitted_pipeline = pickle.load(arguments.fitted_pipeline)

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

    try:
        predictions, pipeline_run = produce(fitted_pipeline, test_inputs)
    except exceptions.PipelineRunError as error:
        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)

    _output_pipeline_runs(arguments, [pipeline_run])


def _score(arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset

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

    try:
        predictions, produce_pipeline_run = produce(fitted_pipeline, test_inputs)
    except exceptions.PipelineRunError as error:
        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)

    try:
        scores, scoring_pipeline_run = score(
            scoring_pipeline,
            fitted_pipeline.problem_description,
            predictions,
            score_inputs,
            metrics,
            context=context,
            random_seed=getattr(arguments, 'random_seed', 0),
            volumes_dir=getattr(arguments, 'volumes_dir', None),
            runtime_environment=runtime_environment,
        )
    except exceptions.PipelineRunError as error:
        if error.pipeline_runs:
            assert len(error.pipeline_runs) == 1, len(error.pipeline_runs)

            # Modifies "produce_pipeline_run" in-place.
            combine_pipeline_runs(
                produce_pipeline_run, scoring_pipeline_run=error.pipeline_runs[0],
            )

        error.pipeline_runs = [produce_pipeline_run]

        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    # Modifies "produce_pipeline_run" in-place.
    combine_pipeline_runs(
        produce_pipeline_run, scoring_pipeline_run=scoring_pipeline_run,
        metrics=metrics, scores=scores, problem_description=fitted_pipeline.problem_description,
    )

    if getattr(arguments, 'scores', None) is not None:
        export_dataframe(scores, arguments.scores)

    _output_pipeline_runs(arguments, [produce_pipeline_run])


def _fit_produce(arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset

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
        problem_description = problem.parse_problem_description(arguments.problem)
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

    if getattr(arguments, 'log_dir', None) is not None:
        log_dir = getattr(arguments, 'log_dir')
        logging.getLogger().setLevel(logging.DEBUG)
        print(f'Logging at directory: {log_dir}')

    try:
        fitted_pipeline, predictions, fit_pipeline_run = fit(
            pipeline, problem_description, inputs, context=context,
            random_seed=getattr(arguments, 'random_seed', 0),
            volumes_dir=getattr(arguments, 'volumes_dir', None),
            runtime_environment=runtime_environment,
            log_dir=log_dir
        )
    except exceptions.PipelineRunError as error:
        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    try:
        predictions, produce_pipeline_run = produce(fitted_pipeline, test_inputs)
    except exceptions.PipelineRunError as error:
        error.pipeline_runs = [fit_pipeline_run] + list(error.pipeline_runs)

        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)

    _output_pipeline_runs(arguments, [fit_pipeline_run, produce_pipeline_run])


def _fit_score(arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset

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
            getattr(arguments, 'datasets_dir', []), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['train_inputs']
        test_inputs = meta['test_inputs']
        score_inputs = meta['score_inputs']
    else:
        problem_description = problem.parse_problem_description(arguments.problem)
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

    try:
        fitted_pipeline, predictions, fit_pipeline_run = fit(
            pipeline, problem_description, inputs, context=context,
            random_seed=getattr(arguments, 'random_seed', 0),
            volumes_dir=getattr(arguments, 'volumes_dir', None),
            runtime_environment=runtime_environment,
        )
    except exceptions.PipelineRunError as error:
        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    try:
        predictions, produce_pipeline_run = produce(fitted_pipeline, test_inputs)
    except exceptions.PipelineRunError as error:
        error.pipeline_runs = [fit_pipeline_run] + list(error.pipeline_runs)

        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)

    try:
        scores, scoring_pipeline_run = score(
            scoring_pipeline, problem_description, predictions, score_inputs, metrics,
            context=context, random_seed=getattr(arguments, 'scoring_random_seed', 0),
            volumes_dir=getattr(arguments, 'volumes_dir', None),
            runtime_environment=runtime_environment,
        )
    except exceptions.PipelineRunError as error:
        if error.pipeline_runs:
            assert len(error.pipeline_runs) == 1, len(error.pipeline_runs)

            # Modifies "produce_pipeline_run" in-place.
            combine_pipeline_runs(
                produce_pipeline_run, scoring_pipeline_run=error.pipeline_runs[0],
            )

        error.pipeline_runs = [fit_pipeline_run, produce_pipeline_run]

        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    # Modifies "produce_pipeline_run" in-place.
    combine_pipeline_runs(
        produce_pipeline_run, scoring_pipeline_run=scoring_pipeline_run,
        metrics=metrics, scores=scores, problem_description=problem_description,
    )

    if getattr(arguments, 'scores', None) is not None:
        export_dataframe(scores, arguments.scores)

    _output_pipeline_runs(arguments, [fit_pipeline_run, produce_pipeline_run])


def _evaluate(arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset

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
        problem_description = problem.parse_problem_description(arguments.problem)
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

    try:
        results_list = evaluate(
            pipeline, data_pipeline, scoring_pipeline, problem_description, inputs, data_params, metrics,
            context=context, random_seed=getattr(arguments, 'random_seed', 0),
            data_random_seed=getattr(arguments, 'data_random_seed', 0),
            scoring_random_seed=getattr(arguments, 'scoring_random_seed', 0),
            volumes_dir=getattr(arguments, 'volumes_dir', None),
            runtime_environment=runtime_environment,
        )
    except exceptions.PipelineRunError as error:
        _output_pipeline_runs(arguments, error.pipeline_runs)

        raise error

    scores_list, fit_pipeline_runs, produce_pipeline_runs = zip(*results_list)

    # "scores_list" is in fact a tuple.
    scores = combine_folds(list(scores_list))

    if getattr(arguments, 'scores', None) is not None:
        export_dataframe(scores, arguments.scores)

    _output_pipeline_runs(arguments, fit_pipeline_runs + produce_pipeline_runs)


def handler(arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
            pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
            dataset_resolver: typing.Callable = None) -> None:
    # Dynamically fetch which subparser was used.
    subparser = parser._subparsers._group_actions[0].choices[arguments.runtime_command]  # type: ignore

    if hasattr(arguments, 'meta'):
        # TODO: These arguments are required, but this is not visible from the usage line. These arguments are marked as optional there.
        manual_config = [('-r/--problem', 'problem'), ('-i/--input', 'inputs'), ('-t/--test-input', 'test_inputs'), ('-a/--score-input', 'score_inputs')]
        if any(hasattr(arguments, dest) and getattr(arguments, dest) is not None for (name, dest) in manual_config) and arguments.meta is not None:
            subparser.error("the following arguments cannot be used together: {manual_arguments} and -m/--meta".format(
                manual_arguments=', '.join(name for (name, dest) in manual_config if hasattr(arguments, dest) and getattr(arguments, dest) is not None),
            ))
        elif any(hasattr(arguments, dest) and getattr(arguments, dest) is None for (name, dest) in manual_config) and arguments.meta is None:
            subparser.error("the following arguments are required: {manual_arguments} or -m/--meta".format(
               manual_arguments=', '.join(name for (name, dest) in manual_config if hasattr(arguments, dest)),
            ))

    # Call a handler for the command.
    arguments.runtime_handler(arguments, pipeline_resolver=pipeline_resolver, meta_parser=meta_parser, dataset_resolver=dataset_resolver)


def configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    if 'random_seed' not in skip_arguments:
        parser.add_argument(
            '-n', '--random-seed', type=int, default=0, action='store', metavar='SEED',
            help="random seed to use",
        )
    if 'context' not in skip_arguments:
        parser.add_argument(
            '-x', '--context', choices=[context.name for context in metadata_base.Context], default=metadata_base.Context.TESTING.name, action='store',
            help="in which context to run pipelines, default is TESTING",
        )
    if 'pipeline_search_paths' not in skip_arguments:
        parser.add_argument(
            '-p', '--pipelines-path', action='append', metavar='PATH', dest='pipeline_search_paths',
            help="path to a directory with pipelines to resolve from (<pipeline id>.json and <pipeline id>.yml), "
                 "can be specified multiple times, has priority over PIPELINES_PATH environment variable",
        )
    if 'volumes_dir' not in skip_arguments:
        parser.add_argument(
            '-v', '--volumes', action='store', dest='volumes_dir',
            help="path to a directory with static files required by primitives, in the standard directory structure (as obtained running \"python3 -m d3m.index download\")",
        )
    if 'datasets_dir' not in skip_arguments:
        parser.add_argument(
            '-d', '--datasets', action='store', dest='datasets_dir',
            help="path to a directory with datasets (and problem descriptions) to resolve IDs in meta files",
        )
    if 'worker_id' not in skip_arguments:
        parser.add_argument(
            '--worker-id', action='store',
            help="globally unique identifier for the machine on which the runtime is running",
        )
    if 'compute_digest' not in skip_arguments:
        parser.add_argument(
            '--compute-digest', choices=[compute_digest.name for compute_digest in dataset_module.ComputeDigest], default=dataset_module.ComputeDigest.ONLY_IF_MISSING.name, action='store',
            help="when loading datasets, when to compute their digests, default is ONLY_IF_MISSING",
        )
    if 'strict_resolving' not in skip_arguments:
        parser.add_argument(
            '--strict-resolving', default=False, action='store_true',
            help="fail resolving if a resolved pipeline or primitive does not fully match specified reference",
        )
    if 'strict_digest' not in skip_arguments:
        parser.add_argument(
            '--strict-digest', default=False, action='store_true',
            help="when loading datasets or pipelines, if computed digest does not match the one provided in metadata, raise an exception?"
        )

    subparsers = parser.add_subparsers(dest='runtime_command', title='commands')
    subparsers.required = True  # type: ignore

    # TODO: Add command to compute "can_accept" over the pipeline.
    fit_parser = subparsers.add_parser(
        'fit', help="fit a pipeline",
        description="Fits a pipeline on train data, resulting in a fitted pipeline. Outputs also produced predictions during fitting on train data.",
    )
    produce_parser = subparsers.add_parser(
        'produce', help="produce using a fitted pipeline",
        description="Produce predictions on test data given a fitted pipeline.",
    )
    score_parser = subparsers.add_parser(
        'score', help="produce using a fitted pipeline and score results",
        description="Produce predictions on test data given a fitted pipeline and compute scores.",
    )
    fit_produce_parser = subparsers.add_parser(
        'fit-produce', help="fit a pipeline and then produce using it",
        description="Fit a pipeline on train data and produce predictions on test data.",
    )
    fit_score_parser = subparsers.add_parser(
        'fit-score', help="fit a pipeline, produce using it and score results",
        description="Fit a pipeline on train data, then produce predictions on test data and compute scores.",
    )
    evaluate_parser = subparsers.add_parser(
        'evaluate', help="evaluate a pipeline",
        description="Run pipeline multiple times using an evaluation approach and compute scores for each run.",
    )

    if 'pipeline' not in skip_arguments:
        fit_parser.add_argument(
            '-p', '--pipeline', action='store', required=True,
            help="path to a pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        fit_parser.add_argument(
            '-r', '--problem', action='store',
            help="path to a problem description file",
        )
    if 'inputs' not in skip_arguments:
        fit_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train dataset",
        )
    if 'meta' not in skip_arguments:
        fit_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'save' not in skip_arguments:
        fit_parser.add_argument(
            '-s', '--save', type=argparse.FileType('wb'), action='store',
            help="save fitted pipeline to a file",
        )
    if 'output' not in skip_arguments:
        fit_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save produced predictions during fitting to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file",
        )
    fit_parser.set_defaults(runtime_handler=_fit)

    if 'fitted_pipeline' not in skip_arguments:
        produce_parser.add_argument(
            '-f', '--fitted-pipeline', type=argparse.FileType('rb'), action='store', required=True,
            help="path to a saved fitted pipeline",
        )
    if 'test_inputs' not in skip_arguments:
        produce_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test dataset",
        )
    if 'meta' not in skip_arguments:
        produce_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'output' not in skip_arguments:
        produce_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save produced predictions to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        produce_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file",
        )
    produce_parser.set_defaults(runtime_handler=_produce)

    if 'fitted_pipeline' not in skip_arguments:
        score_parser.add_argument(
            '-f', '--fitted-pipeline', type=argparse.FileType('rb'), action='store', required=True,
            help="path to a saved fitted pipeline",
        )
    if 'scoring_pipeline' not in skip_arguments:
        score_parser.add_argument(
            '-n', '--scoring-pipeline', action='store', required=True,
            help="path to a scoring pipeline file (.json or .yml) or pipeline ID",
        )
    if 'test_inputs' not in skip_arguments:
        score_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test dataset",
        )
    if 'score_inputs' not in skip_arguments:
        score_parser.add_argument(
            '-a', '--score-input', action='append', metavar='INPUT', dest='score_inputs',
            help="path or URI of an input score dataset",
        )
    if 'meta' not in skip_arguments:
        score_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'metrics' not in skip_arguments:
        score_parser.add_argument(
            '-e', '--metric', choices=[metric.name for metric in problem.PerformanceMetric], action='append', metavar='METRIC', dest='metrics',
            help="metric to use, using default parameters, can be specified multiple times, default from problem description",
        )
    if 'output' not in skip_arguments:
        score_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions to a file",
        )
    if 'scores' not in skip_arguments:
        score_parser.add_argument(
            '-c', '--scores', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        score_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file",
        )
    score_parser.set_defaults(runtime_handler=_score)

    if 'pipeline' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-p', '--pipeline', action='store', required=True,
            help="path to a pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-r', '--problem', action='store',
            help="path to a problem description file",
        )
    if 'inputs' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train dataset",
        )
    if 'test_inputs' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test dataset",
        )
    if 'meta' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'save' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-s', '--save', type=argparse.FileType('wb'), action='store',
            help="save fitted pipeline to a file",
        )
    if 'output' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save produced predictions to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file",
        )
    if 'log_dir' not in skip_arguments:
        fit_produce_parser.add_argument(
            '--log-dir', default=None, action='store',
            help="set logging directory and set logging level to debug"
        )
    fit_produce_parser.set_defaults(runtime_handler=_fit_produce)

    if 'pipeline' not in skip_arguments:
        fit_score_parser.add_argument(
            '-p', '--pipeline', action='store', required=True,
            help="path to a pipeline file (.json or .yml) or pipeline ID",
        )
    if 'scoring_pipeline' not in skip_arguments:
        fit_score_parser.add_argument(
            '-n', '--scoring-pipeline', action='store', required=True,
            help="path to a scoring pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        fit_score_parser.add_argument(
            '-r', '--problem', action='store',
            help="path to a problem description file",
        )
    if 'inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train dataset",
        )
    if 'test_inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test dataset",
        )
    if 'score_inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-a', '--score-input', action='append', metavar='INPUT', dest='score_inputs',
            help="path or URI of an input score dataset",
        )
    if 'meta' not in skip_arguments:
        fit_score_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'metrics' not in skip_arguments:
        fit_score_parser.add_argument(
            '-e', '--metric', choices=[metric.name for metric in problem.PerformanceMetric], action='append', metavar='METRIC', dest='metrics',
            help="metric to use, using default parameters, can be specified multiple times, default from problem description",
        )
    if 'save' not in skip_arguments:
        fit_score_parser.add_argument(
            '-s', '--save', type=argparse.FileType('wb'), action='store',
            help="save fitted pipeline to a file",
        )
    if 'output' not in skip_arguments:
        fit_score_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions to a file",
        )
    if 'scores' not in skip_arguments:
        fit_score_parser.add_argument(
            '-c', '--scores', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_score_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file",
        )
    if 'scoring_random_seed' not in skip_arguments:
        fit_score_parser.add_argument(
            '--scoring-random-seed', type=int, action='store', default=0,
            help="random seed to use for scoring",
        )
    fit_score_parser.set_defaults(runtime_handler=_fit_score)

    if 'pipeline' not in skip_arguments:
        evaluate_parser.add_argument(
            '-p', '--pipeline', action='store', required=True,
            help="path to a pipeline file (.json or .yml) or pipeline ID"
        )
    if 'data_pipeline' not in skip_arguments:
        evaluate_parser.add_argument(
            '-d', '--data-pipeline', action='store', required=True,
            help="path to a data preparation pipeline file (.json or .yml) or pipeline ID",
        )
    if 'scoring_pipeline' not in skip_arguments:
        evaluate_parser.add_argument(
            '-n', '--scoring-pipeline', action='store', required=True,
            help="path to a scoring pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        evaluate_parser.add_argument(
            '-r', '--problem', action='store',
            help="path to a problem description file",
        )
    if 'inputs' not in skip_arguments:
        evaluate_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input full dataset",
        )
    if 'meta' not in skip_arguments:
        evaluate_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'data_params' not in skip_arguments:
        evaluate_parser.add_argument(
            '-y', '--data-param', nargs=2, action='append', metavar=('NAME', 'VALUE'), dest='data_params',
            help="hyper-parameter name and its value for data preparation pipeline, can be specified multiple times, value should be JSON-serialized",
        )
    if 'data_split_file' not in skip_arguments:
        evaluate_parser.add_argument(
            '--data-split-file', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="reads the split file and populates \"primary_index_values\" hyper-parameter for data preparation pipeline with values from the \"d3mIndex\" column corresponding to the test data",
        )
    if 'metrics' not in skip_arguments:
        evaluate_parser.add_argument(
            '-e', '--metric', choices=[metric.name for metric in problem.PerformanceMetric], action='append', metavar='METRIC', dest='metrics',
            help="metric to use, using default parameters, can be specified multiple times, default from problem description",
        )
    if 'scores' not in skip_arguments:
        evaluate_parser.add_argument(
            '-c', '--scores', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        evaluate_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file",
        )
    if 'data_random_seed' not in skip_arguments:
        evaluate_parser.add_argument(
            '--data-random-seed', type=int, action='store', default=0,
            help="random seed to use for data preparation",
        )
    if 'scoring_random_seed' not in skip_arguments:
        evaluate_parser.add_argument(
            '--scoring-random-seed', type=int, action='store', default=0,
            help="random seed to use for scoring",
        )
    evaluate_parser.set_defaults(runtime_handler=_evaluate)


def main() -> None:
    logging.basicConfig()

    parser = argparse.ArgumentParser(description="Run D3M pipelines with default hyper-parameters.")
    configure_parser(parser)

    arguments = parser.parse_args()

    handler(arguments, parser)


if __name__ == '__main__':
    main()
