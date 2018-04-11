import uuid
import copy
import threading

import numpy as np

from abc import ABC
from collections import defaultdict
from typing import List

from dsbox.schema.problem_schema import Metric

class CrossValidationStat(object):
    """Record score for each metric and for each fold."""
    def __init__(self):
        self.fold_metric_values = defaultdict(list)

    def add_fold_metric(self, metric: Metric, value):
        '''Add validation score from one cross validation fold'''
        self.fold_metric_values[metric].append(value)

    def get_metric(self, metric: Metric):
        '''Returns mean value for metric across each cross-validation fold'''
        return sum(self.fold_metric_values[metric]) / len(self.fold_metric_values[metric])

    def get_standard_error(self, metric: Metric):
        '''Returns standard error for metric across each cross-validation fold'''
        print(self.fold_metric_values[metric])
        vals = np.asarray(self.fold_metric_values[metric])
        return np.sqrt(np.var(vals)/len(vals))

class PipelineExecutionResult(object):
    """
    Defines a pipeline execution result
    """
    def __init__(self, predictions, metric_values, stat: CrossValidationStat):
        self.predictions = predictions # Predictions dataframe
        self.metric_values = metric_values # Dictionary of metric to value
        self.stat = stat

    def get_value(self, metric: Metric):
        return self.metric_values[metric.name]

class Pipeline(object):
    """
    Defines a pipeline
    """
    def __init__(self, id=None, primitives=None, ensemble= None):
        if id is None:
            id = str(uuid.uuid4())

        if primitives is None:
            primitives = []
        self.id = id
        self.primitives = primitives

        # Execution Results
        self.planner_result: PipelineExecutionResult = None
        self.test_result: PipelineExecutionResult = None

        # Ensemble?
        self.ensemble = ensemble


        # Change notification
        self.changes = threading.Condition()
        self.finished = False

    def clone(self, idcopy=False):
        pipeline = copy.deepcopy(self)
        if not idcopy:
            pipeline.id = str(uuid.uuid4())
        return pipeline

    def setPipelineId(self, id):
        self.id = id

    def setPrimitives(self, primitives):
        self.primitives = primitives

    def addPrimitive(self, primitive):
        self.primitives.append(primitive)

    def length(self):
        return len(self.primitives)

    def getPrimitiveAt(self, index):
        return self.primitives[index]

    def insertPrimitiveAt(self, index, primitive):
        self.primitives.insert(index, primitive)

    def replacePrimitiveAt(self, index, primitive):
        self.primitives[index] = primitive

    def replaceSubpipelineAt(self, index, subpipeline):
        self.primitives[index:index] = subpipeline.primitives

    def notifyChanges(self):
        self.changes.acquire()
        self.changes.notifyAll()
        self.changes.release()

    def waitForChanges(self):
        self.changes.acquire()
        self.changes.wait()
        self.changes.release()

    def getLearnerExecutableSize(self):
        if self.ensemble:
            size = 0
            for pipeline in self.ensemble.all_pipelines:
                size += pipeline.getPrimitiveAt(-1).getExecutableSize()
        else:
            size = self.primitives[-1].getExecutableSize()
        return size

    def __str__(self):
        if self.ensemble:
            return str(self.ensemble.pipelines)
        else:
            return str(self.primitives)

    def __repr__(self):
        if self.ensemble:
            return str(self.ensemble.pipelines)
        else:
            return str(self.primitives)

    def __getstate__(self):
        return (self.id, self.primitives, self.planner_result, self.test_result, self.finished, self.ensemble)

    def __setstate__(self, state):
        self.id, self.primitives, self.planner_result, self.test_result, self.finished, self.ensemble= state
        self.changes = threading.Condition()

class PipelineSorter(ABC):

    def __init__(self, metric):
        self.metric: Metric = metric

    def sort_pipelines(self, pipelines: List[Pipeline]):
        pass

class MetricPipelineSorter(PipelineSorter):
    '''Sort pipelines strictly based on metric score'''

    def __init__(self, metric):
        super().__init__(metric)

    def sort_pipelines(self, pipelines):
        return sorted(pipelines, key=lambda p: p.planner_result.get_value(self.metric),
                      reverse=self.metric.larger_is_better())

class OneStandardErrorPipelineSorter(PipelineSorter):
    '''Sort pipelines based on one-standard-error rule.

    One-standard-error model selection rule selects the small size
    model among those models that are within one standard error of the
    best model'''

    def __init__(self, metric):
        super().__init__(metric)

    def sort_pipelines(self, pipelines):
        if len(pipelines) <= 1:
            return pipelines[:]

        metric_sorted = MetricPipelineSorter(self.metric).sort_pipelines(pipelines)
        best_result = metric_sorted[0].planner_result
        best_value = best_result.get_value(self.metric)

        if best_result.stat is None:
            # Best result is an ensemble.
            # TODO: What should be the standard error? For now just use the standard error from next best pipeline
            std_error = metric_sorted[1].planner_result.stat.get_standard_error(self.metric)
        else:
            std_error = best_result.stat.get_standard_error(self.metric)
        close = []
        rest = []
        for pipe in metric_sorted:
            delta = abs(pipe.planner_result.get_value(self.metric) - best_value)
            if delta < std_error:
                close.append(pipe)
            else:
                rest.append(pipe)
        close = sorted(close, key=lambda p: p.getLearnerExecutableSize())
        all = close + rest

        for pipeline in all:
            metric_values = []
            for metric in pipeline.planner_result.metric_values.keys():
                metric_value = pipeline.planner_result.metric_values[metric]
                metric_values.append("%s = %2.4f" % (metric, metric_value))
            metric_values.append('executable_size = {}'.format(pipeline.getLearnerExecutableSize()))
            print("%s ( %s ) : %s\n" % (pipeline.id, pipeline, metric_values))

        return all
