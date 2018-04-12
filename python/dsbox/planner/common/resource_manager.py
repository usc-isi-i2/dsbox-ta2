'''Resource Manager for pipelines'''
import asyncio
import concurrent
import copy
import json
import logging
import sys
import multiprocessing
import traceback

from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Dict, List

import numpy as np
import stopit

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.externals import joblib

from dsbox.schema.data_profile import DataProfile
from dsbox.planner.common.pipeline import Pipeline, PipelineExecutionResult
from dsbox.planner.common.primitive import Primitive
from dsbox.executer.executionhelper import ExecutionHelper

TIMEOUT = 600  # Time out primitives running for more than 10 minutes

# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(name)s: %(message)s')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')

class MyExecutor(concurrent.futures.ProcessPoolExecutor):
    '''Used to generate debugging prints'''
    def __init__(self, max_workers=2):
        # print('max_workers={}'.format(max_workers))
        super().__init__(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        # print('executor submit({})'.format(fn))
        return super().submit(fn, *args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return super().map(fn, *iterables, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait=True):
        super().shutdown(wait=wait)

class PipelineExecStat:
    '''Pipeline execution statistics'''
    def __init__(self, pipeline, pending_at=None):
        self.pipeline = pipeline
        self.pending_at = pending_at if pending_at else datetime.now()
        self.running_at = None
        self.finishing_at = None
    def done(self):
        '''Returns True if pipeline finished'''
        return self.finishing_at is not None
    def get_running_time(self) -> timedelta:
        '''Returns pipeline wallclock running time'''
        return self.finishing_at - self.running_at
    def __str__(self):
        now = datetime.now()
        if self.running_at is None:
            return 'pending {:.1f} min'.format((now-self.pending_at)/timedelta(seconds=60))
        elif self.finishing_at is None:
            return 'running {:.1f} min'.format((now-self.running_at)/timedelta(seconds=60))
        return 'finished {:.1f} min'.format((self.finishing_at-self.running_at)/timedelta(seconds=60))

class PrimitiveExecStat:
    '''primitive execution statistics'''
    def __init__(self, primitive, use_cache=False, pending_at=None):
        self.primitive = primitive
        self.use_cache = use_cache
        self.pending_at = pending_at if pending_at else datetime.now()
        self.finishing_at = None
    def done(self):
        '''Returns True if pipeline finished'''
        return self.finishing_at is not None
    def get_running_time(self) -> timedelta:
        '''Returns primitive wallclock running time'''
        return self.finishing_at - self.pending_at
    def __str__(self):
        now = datetime.now()
        if self.finishing_at is None:
            if self.use_cache:
                return 'waiting {:.1f} min'.format((now-self.pending_at)/timedelta(seconds=60))
            else:
                return 'running {:.1f} min'.format((now-self.pending_at)/timedelta(seconds=60))
        return 'finished {:.1f} min'.format((self.finishing_at-self.pending_at)/timedelta(seconds=60))


class SimpleEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, (ClassifierMixin, RegressorMixin)):
            return str(type(obj))
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class ExecutionStatistics:
    '''Execution status of all pipelines and their primitives'''
    def __init__(self):
        # For each pipeline, store its statistics
        self.pipeline_stats = dict()  # type: Dict[str, PipelineExecStat]

        # For each pipeline, store statistics for its primitve statistics
        self.primitives = defaultdict(list)  # type: Dict[str, List[PrimitiveExecStat]]

        # For unfinished pipeline statistics
        self.unfinished_pipelines = dict()

        self.num_pipelines = 0
        self.num_pipelines_finished = 0
        self.num_pipelines_successful = 0

        self.starting_at = datetime.now()
        self.ending_at = None

        self._encoder = SimpleEncoder()

    def pipeline_pending(self, pipeline: Pipeline):
        '''Pipeline submitted for execution'''
        if pipeline.id in self.pipeline_stats:
            print('Pipeline already registerd: {} {}', pipeline.id, pipeline)
        else:
            self.num_pipelines += 1
            self.pipeline_stats[pipeline.id] = PipelineExecStat(pipeline)
            self.unfinished_pipelines[pipeline.id] = pipeline

    def pipeline_running(self, pipeline: Pipeline):
        '''Pipeline started running'''
        self.pipeline_stats[pipeline.id].running_at = datetime.now()

        # Replace with this pipeline, because now we have the execution version of the pipeline
        self.pipeline_stats[pipeline.id].pipeline = pipeline

    def pipeline_finished(self, pipeline: Pipeline):
        '''Pipeline finished running'''
        self.num_pipelines_finished += 1
        self.pipeline_stats[pipeline.id].finishing_at = datetime.now()
        del self.unfinished_pipelines[pipeline.id]

        if pipeline.planner_result is not None:
            self.num_pipelines_successful += 1

        # Replace with this pipeline, because now we have the execution version of the pipeline
        self.pipeline_stats[pipeline.id].pipeline = pipeline

    def primitive_waiting(self, pipeline: Pipeline, primitive: Primitive):
        '''Primitive started waiting for results cached from another primitive'''
        # Primitives can either use the result cache
        self.primitives[pipeline.id].append(PrimitiveExecStat(primitive, use_cache=True))

    def primitive_running(self, pipeline: Pipeline, primitive: Primitive):
        '''Primitive started running'''
        # Or, primitives run to get their own results
        self.primitives[pipeline.id].append(PrimitiveExecStat(primitive, use_cache=False))

    def primitive_finishing(self, pipeline: Pipeline, primitive: Primitive):
        '''Primitive got cached result, or finished running'''
        self.primitives[pipeline.id][-1].finishing_at = datetime.now()

    def print_status(self):
        '''prints current state'''
        now = datetime.now()
        print('Status {}:'.format(now.isoformat()))
        print('Sucessful pipelines of the finished pipelines: {} of {}'.format(
            self.num_pipelines_successful, self.num_pipelines_finished))
        print('Unfinished pipelines of all pipelines: {} of  {}:'.format(
            self.num_pipelines-self.num_pipelines_finished, self.num_pipelines))
        for pipeline in self.unfinished_pipelines.values():
            stat = self.pipeline_stats[pipeline.id]
            print(' Pipeline {} {} {}'.format(pipeline.id, stat, pipeline))
        print('Unfinished primtives:')
        for pipeline in self.unfinished_pipelines.values():
            if self.primitives[pipeline.id]:
                stat = self.primitives[pipeline.id][-1]
                print(' Primitive {} {} {}'.format(pipeline.id, stat, stat.primitive))
        sys.stdout.flush()

    def print_successful_pipelines(self):
        '''Print sucessful pipelines'''
        pipeline_stats: PipelineExecStat = self.pipeline_stats.values()
        pipelines = [s.pipeline for s in pipeline_stats if s.pipeline.planner_result is not None]
        metrics = [name for name in pipelines[0].planner_result.metric_values.keys()]
        pipelines_sorted = sorted(pipelines, key=lambda p: p.planner_result.metric_values[metrics[0]], reverse=True)
        for pipeline in pipelines_sorted:
            metric_values = []
            for metric in pipeline.planner_result.metric_values.keys():
                metric_value = pipeline.planner_result.metric_values[metric]
                metric_values.append("%s = %2.4f" % (metric, metric_value))
            print("%s ( %s ) : %s" % (pipeline.id, pipeline, metric_values))

    def pickle_without_executables(self, filename):
        '''Save statistics. WARNING: will remove primitive executables to avoid large files'''
        for pipeline_stat in self.pipeline_stats.values():
            for primitive in pipeline_stat.pipeline.primitives:
                primitive.executables = None
        joblib.dump(self, filename)

    def get_stat_by_model(self, name):
        '''Returns pipeline stat by learner name'''
        for stat in self.pipeline_stats.values():
            if name in stat.pipeline.primitives[-1].name:
                yield stat

    def json_line_dump(self, out=sys.stdout, *, run_id=None, problem_id='a_run', dataset_names=['a_dataset']):
        '''Dump data out in JSON Line format'''
        if self.ending_at is None:
            self.ending_at = datetime.now()
        if run_id is None:
            run_id = datetime.now().isoformat()
        run_info = {
            'run_info' : True,
            'run_id' : run_id,
            'problem_id' : problem_id,
            'dataset' : dataset_names,
            'num_pipelines' : self.num_pipelines,
            'num_pipelines_finished' : self.num_pipelines_finished,
            'num_pipelines_successful' : self.num_pipelines_successful,
            'running' : self.starting_at,
            'finishing' : self.ending_at,
            'running_time' : (self.ending_at - self.starting_at).total_seconds()
        }
        print(self._encoder.encode(run_info), file=out)

        for pipe_id, pipe_stat in self.pipeline_stats.items():
            primitives_info = []
            for primitive_stat in self.primitives[pipe_id]:
                if isinstance(primitive_stat.primitive, str):
                    primitive = {
                        'id' : primitive_stat.primitive,
                        'name' : primitive_stat.primitive,
                        'class' : primitive_stat.primitive,
                        'hyperparams' : None}
                else:
                    primitive = {
                        'id' : primitive_stat.primitive.id,
                        'name' : primitive_stat.primitive.name,
                        'class' : primitive_stat.primitive.cls,
                        'hyperparams' : (primitive_stat.primitive.getHyperparams()
                                         if primitive_stat.primitive.hasHyperparamClass() else None)}
                primitives_info.append({
                    **primitive,
                    'pending' : primitive_stat.pending_at if primitive_stat.pending_at else None,
                    'finishing' : primitive_stat.finishing_at if primitive_stat.finishing_at else None,
                    'done' : primitive_stat.done(),
                    'running_time' : (primitive_stat.get_running_time().total_seconds()
                                      if primitive_stat.done() else None),
                    'use_cache': primitive_stat.use_cache
                })
            pipe_info = {
                'pipe_info' : True,
                'run_id' : run_id,
                'problem_id' : problem_id,
                'dataset' : dataset_names,
                'pipe_id' : pipe_stat.pipeline.id,
                'training_metric': (pipe_stat.pipeline.planner_result.metric_values
                                    if pipe_stat.pipeline.planner_result else None),
                'pending' : pipe_stat.pending_at if pipe_stat.pending_at else None,
                'running' : pipe_stat.running_at if pipe_stat.running_at else None,
                'finishing' : pipe_stat.finishing_at if pipe_stat.finishing_at else None,
                'done' : pipe_stat.done(),
                'running_time' : pipe_stat.get_running_time().total_seconds() if pipe_stat.done() else None,
                'primitives' : primitives_info}
            print(self._encoder.encode(pipe_info), file=out)

class ResourceManager:
    '''Resource manager for running pipelines.

    Use asyncio event loop to schedule primitive executions. The
    actual execution of primitives can be inline or within a subprocess.

    '''
    def __init__(self, helper: ExecutionHelper, max_workers=0):
        self.helper = helper
        self.log = logging.getLogger('ResourceManager')
        self.loop = asyncio.new_event_loop()
        self.loop.set_debug(enabled=True)
        self.loop.set_exception_handler(self._exception_handler)

        # Pipelines that have been executed
        self.exec_pipelines = []

        # Cache results of execution
        self.execution_cache = {}

        # Cache trained primitive executables
        self.primitive_cache = {}

        # Prmitives scheduled for execution
        self.scheduled = set()

        # Tasks to run
        self.pending_tasks = []

        # Used by primitives waiting for results
        self.condition = dict()

        # Set max number of subprocesses
        if max_workers == 0:
            max_workers = multiprocessing.cpu_count()
        # self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        self.executor = MyExecutor(max_workers=max_workers)

        self.cross_validation_folds = 10

        self.stats = ExecutionStatistics()

    @stopit.threading_timeoutable()
    def execute_pipelines(self, pipelines, df, df_lbl, callbacks=None):
        """Execute all pipelines.

        Blocking returns when they are all pipelines complete
        (including add_pipelines).

        """
        # start status reporting task
        status_task = self.loop.create_task(self._report_status())

        # Create and schedule tasks for each pipeline
        tasks = []
        for pipeline in pipelines:
            tasks.append(self.loop.create_task(self._run_pipeline(pipeline, df, df_lbl)))
            self.stats.pipeline_pending(pipeline)

        if callbacks is not None:
            for task, callback in zip(tasks, callbacks):
                task.add_done_callback(callback)

        # Run pipeline tasks
        try:
            self.log.info('Run initial pipelines')
            self.loop.run_until_complete(asyncio.gather(*tasks))
            self.stats.print_status()

            # Make sure pipelines add later are completed
            self.pending_tasks = [task for task in self.pending_tasks if not task.done()]

            while len(self.pending_tasks) > 0:
                self.log.info('Run pending pipelines %d', len(self.pending_tasks))
                self.loop.run_until_complete(asyncio.gather(*self.pending_tasks))
                self.pending_tasks = [task for task in self.pending_tasks if not task.done()]

        except Exception as e:
            print('Exception in running pipelines: {}'.format(e))
            traceback.print_exc()

        status_task.cancel()
        self.loop.run_until_complete(status_task)

    def add_pipeline(self, pipeline, df, df_lbl, callback=None):
        '''Add one pipeline for execution.

        Non-blocking returns right away.  Assumes execute_pipelines
        has been called or will be called

        '''
        self.log.debug('Adding pipeline %s %s', pipeline.id, pipeline)

        # Create and schedule task
        task = self.loop.create_task(self._run_pipeline(pipeline, df, df_lbl))

        self.stats.pipeline_pending(pipeline)

        if callback is not None:
            task.add_done_callback(callback)

        # Make sure this task will be ran
        self.pending_tasks.append(task)

    def _exception_handler(self, loop, context):
        print('my_handler: {}'.format(context['message']), file=sys.stderr)
        print('{}'.format(context), file=sys.stderr)

    async def _report_status(self):
        try:
            while True:
                self.stats.print_status()
                await asyncio.sleep(30)
        except concurrent.futures.CancelledError:
            self.stats.print_status()
            return

    async def _run_pipeline(self, pipeline, df, df_lbl):
        print("** Running Pipeline: %s %s" % (pipeline.id, pipeline))
        sys.stdout.flush()
        self.log.debug('%s Pipeline running %s', pipeline.id, pipeline)

        exec_pipeline = pipeline.clone(idcopy=True)
        exec_pipeline.planner_result = None

        self.stats.pipeline_running(exec_pipeline)

        cachekey = ""

        for primitive in exec_pipeline.primitives:

            # Mark the pipeline that the primitive is part of
            # - Used to notify waiting threads of execution changes
            primitive.pipeline = exec_pipeline

            # Include hyperparameter in cache key
            cachekey = "%s.%s" % (cachekey, primitive)

            # Check if result is in cache
            if cachekey in self.execution_cache:
                self.log.debug('%s Primitive cache for %s', pipeline.id, primitive)
                self.stats.primitive_waiting(exec_pipeline, primitive)

                df = self.execution_cache.get(cachekey)
                (primitive.executables, primitive.unified_interface) = self.primitive_cache.get(cachekey)

                self.stats.primitive_finishing(exec_pipeline, primitive)
                continue

            # Check if it is already being processed
            if cachekey in self.scheduled:
                self.log.debug('%s Primitive waiting condition for %s', pipeline.id, primitive)
                self.stats.primitive_waiting(exec_pipeline, primitive)

                if not cachekey in self.condition:
                    self.condition[cachekey] = asyncio.Condition()
                cv = self.condition[cachekey]
                with await cv:
                    await cv.wait()
                self.log.debug('%s Primitive wait condition for %s', pipeline.id, primitive)
                df = self.execution_cache.get(cachekey)
                (primitive.executables, primitive.unified_interface) = self.primitive_cache.get(cachekey)

                self.stats.primitive_finishing(exec_pipeline, primitive)
                continue

            # Run the primitive
            try:
                if df is None:
                    # primitive in previous stage failed
                    self.log.debug('%s Primitive previous stage failed %s', pipeline.id, primitive)
                    return None

                self.log.debug('%s Primitive scheduling %s', pipeline.id, cachekey)

                self.scheduled.add(cachekey)
                await self._run_primitive(exec_pipeline, cachekey, primitive, df, df_lbl)
                self.scheduled.remove(cachekey)
                self.log.debug('%s Primitive done %s', pipeline.id, cachekey)

                df = self.execution_cache.get(cachekey)
                (primitive.executables, primitive.unified_interface) = self.primitive_cache.get(cachekey)

                # Notify condition
                if cachekey in self.condition:
                    self.log.debug('%s notify condition %s', pipeline.id, cachekey)
                    cv = self.condition[cachekey]
                    del self.condition[cachekey]
                    with await cv:
                        cv.notify_all()
            except Exception as e:
                sys.stderr.write(
                    "ERROR execute_pipeline(%s %s) : %s\n" % (exec_pipeline, exec_pipeline.id, e))
                traceback.print_exc()
                exec_pipeline.finished = True
                return None

        self.stats.pipeline_finished(exec_pipeline)

        # Add to the list of executable pipelines
        if exec_pipeline.planner_result is not None:
            self.exec_pipelines.append(exec_pipeline)
            self.log.info('%s Pipeline suceeded %s', exec_pipeline.id, exec_pipeline)
            self.log.info('%s %s %s', exec_pipeline.id, exec_pipeline, exec_pipeline.planner_result.metric_values)
            self.log.info('Number of exec_pipelines = %d', len(self.exec_pipelines))

            return exec_pipeline
        self.log.debug('%s Pipeline failed %s', pipeline.id, pipeline)
        return None


    async def _run_primitive(self, exec_pipeline, cachekey, primitive, df, df_lbl):
        '''Run one primitive'''
        inline = getattr(primitive, 'run_inline', False)

        self.log.debug('Running inline %s primitive %s', inline, primitive)

        executables = None
        unified_interface = False
        if primitive.task == "FeatureExtraction":
            # Featurisation Primitive
            self.log.debug('%s Run primitive feature   %s', exec_pipeline.id, primitive)
            if inline:
                self.stats.primitive_running(exec_pipeline, primitive)
                df = self.helper.featurise(primitive, copy.copy(df), timeout=TIMEOUT)
                self.stats.primitive_finishing(exec_pipeline, primitive)
            else:
                self.stats.primitive_running(exec_pipeline, primitive)
                self.log.debug('%s Run primitive submit    %s', exec_pipeline.id, primitive)
                task = self.loop.run_in_executor(self.executor, self.helper.featurise_remote, primitive, df)
                self.log.debug('%s Run primitive waiting   %s', exec_pipeline.id, primitive)
                await asyncio.wait([task], timeout=TIMEOUT)
                self.log.debug('%s Run primitive wait done %s', exec_pipeline.id, primitive)
                self.stats.primitive_finishing(exec_pipeline, primitive)

                if task.done():
                    result = task.result()
                    if isinstance(result, Exception):
                        self.log.debug(result)
                        df = None
                    else:
                        df, executables, unified_interface = result
                else:
                    df = None

                primitive.executables = executables
                primitive.unified_interface = unified_interface
            self.primitive_cache[cachekey] = (primitive.executables, primitive.unified_interface)
            self.execution_cache[cachekey] = df
        elif primitive.task == "Modeling":
            # Modeling Primitive
            self.log.debug('%s Run primitive modeling  %s', exec_pipeline.id, primitive)

            # always run in subprocess
            self.stats.primitive_running(exec_pipeline, primitive)
            self.log.debug('%s Run primitive submit    %s', exec_pipeline.id, primitive)
            task = self.loop.run_in_executor(self.executor, self.helper.cross_validation_score,
                                             primitive, df, df_lbl, self.cross_validation_folds)

            self.log.debug('%s Run primitive waiting   %s', exec_pipeline.id, primitive)
            await asyncio.wait([task], timeout=TIMEOUT)
            self.log.debug('%s Run primitive wait done %s', exec_pipeline.id, primitive)
            self.stats.primitive_finishing(exec_pipeline, primitive)

            # Get model predictions and metric values.
            metric_values = []
            if task.done():
                result = task.result()
                if not isinstance(result, Exception) and result is not None:
                    predictions, metric_values, cross_validation_stat = result

            if metric_values and len(metric_values) > 0:
                print("Got results from %s" % exec_pipeline)
                self.log.debug("%s Got results from %s", exec_pipeline.id, primitive)

                self.stats.primitive_running(exec_pipeline, 'cross_validation')
                self.log.debug('%s Run primitive submit     cross validation for %s', exec_pipeline.id, primitive)
                task = self.loop.run_in_executor(self.executor, self.helper.create_primitive_model_remote,
                                                 primitive, df, df_lbl)
                self.log.debug('%s Run primitive waiting   cross validation for %s', exec_pipeline.id, primitive)
                await asyncio.wait([task], timeout=TIMEOUT)
                self.log.debug('%s Run primitive wait done cross validation for %s', exec_pipeline.id, primitive)
                self.stats.primitive_finishing(exec_pipeline, 'cross_validation')

                if task.done():
                    result = task.result()
                    if isinstance(result, Exception):
                        print('Exception')
                        print(result)
                        executables = None
                    else:
                        executables, unified_interface = result
                else:
                    self.log.debug('%s Run primitive NOT DONE  cross validation for %s', exec_pipeline.id, primitive)

                # Store the execution result
                exec_pipeline.planner_result = PipelineExecutionResult(predictions, metric_values, cross_validation_stat)

                # Cache the primitive instance
            primitive.executables = executables
            primitive.unified_interface = unified_interface
            self.primitive_cache[cachekey] = (primitive.executables, primitive.unified_interface)

        else:
            self.log.debug('%s Run primitive other primitive type', exec_pipeline.id)
            if inline:
                # Re-profile intermediate data here.
                # TODO: Recheck if it is ok for the primitive's preconditions
                #       and patch pipeline if necessary
                cur_profile = DataProfile(df)

                # Glue primitive
                df = self.helper.execute_primitive(
                    primitive, copy.copy(df), df_lbl, cur_profile, timeout=TIMEOUT)
                self.primitive_cache[cachekey] = (primitive.executables, primitive.unified_interface)
            else:
                cur_profile = DataProfile(df)

                self.log.debug('%s Run primitive submit    %s', exec_pipeline.id, primitive)
                task = self.loop.run_in_executor(self.executor, self.helper.execute_primitive_remote, primitive,
                                                 df, df_lbl, cur_profile)
                self.log.debug('%s Run primitive waiting   %s', exec_pipeline.id, primitive)
                await asyncio.wait([task], timeout=TIMEOUT)
                self.log.debug('%s Run primitive wait done %s', exec_pipeline.id, primitive)

                if task.done():
                    result = task.result()
                    if isinstance(result, Exception):
                        self.log.debug(result)
                        df = None
                    else:
                        df, executables, unified_interface = result
                else:
                    df = None
                primitive.executables = executables
                primitive.unified_interface = unified_interface
            self.primitive_cache[cachekey] = (primitive.executables, primitive.unified_interface)
            self.execution_cache[cachekey] = df
