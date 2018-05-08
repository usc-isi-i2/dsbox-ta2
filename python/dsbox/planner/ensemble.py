import os
import sys
import os.path
import uuid
import copy
import math
import json
import numpy as np
import shutil
import traceback
import inspect
import importlib
import pandas as pd
import time

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.planner.common.pipeline import Pipeline, PipelineExecutionResult
from dsbox.planner.common.resource_manager import ResourceManager
from dsbox.planner.common.problem_manager import Metric, TaskType, TaskSubType
from sklearn.model_selection import KFold
from dsbox.planner.common.pipeline import CrossValidationStat

MIN_METRICS = [Metric.MEAN_SQUARED_ERROR, Metric.ROOT_MEAN_SQUARED_ERROR, Metric.ROOT_MEAN_SQUARED_ERROR_AVG, Metric.MEAN_ABSOLUTE_ERROR, Metric.EXECUTION_TIME]
DISCRETE_METRIC = [TaskSubType.BINARY, TaskSubType.MULTICLASS, TaskSubType.MULTILABEL, TaskSubType.OVERLAPPING, TaskSubType.NONOVERLAPPING]

class Ensemble(object):
    def __init__(self, problem, median = True, max_pipelines = 5):
        self.max_pipelines = max_pipelines
        self.predictions = None  # Predictions dataframe
        self.metric_values =  None # Dictionary of metric to value
        self.train_result = None
        self.test_result = None
        self.score = 0
        self.all_pipelines = []
        #self.test_pipeline_ids = []
        self.problem = problem
        self.median = median
        self._analyze_metrics()
        self.prediction_range = [-np.inf, np.inf]

    def _analyze_metrics(self):
        # *** ONLY CONSIDERS 1 METRIC ***
        #self.minimize_metric = True if self.problem.metrics[0] in MIN_METRICS else False
        self.minimize_metric = []
        for i in range(0, len(self.problem.metrics)):
            print(self.problem.metrics[i])
            self.minimize_metric.append(True if self.problem.metrics[i] in MIN_METRICS else False)
        self.discrete_metric = True if self.problem.task_subtype in DISCRETE_METRIC else False


    def greedy_add(self, pipelines, X, y, max_pipelines = None, plan = True, median = None, pred_minmax_stdevs = 2, cv = 10, seed = 0):
        self.median = median if median is not None else self.median

        stdev_pred = np.std(y.values)
        self.prediction_range = [np.min(y.values) - pred_minmax_stdevs*stdev_pred, np.max(y.values) + pred_minmax_stdevs*stdev_pred]
        print("Using MEDIAN for ENSEMBLE Add" if self.median else "Using MEAN for ENSEMBLE Add")
        which_result = 'planner_result' if plan else 'test_result'
        tic = time.time()
        if self.predictions is None:
            self.predictions = pd.DataFrame(index = X.index, columns = y.columns).fillna(0)
            self.all_pipelines = []

        max_pipelines = self.max_pipelines if max_pipelines is None else max_pipelines
        found_improvement = True
        
        while found_improvement and len(np.unique([pl.id for pl in self.all_pipelines])) < max_pipelines:
            best_score =  float('inf') if self.minimize_metric else 0
            if self.metric_values is not None:
                best_metrics = self.metric_values

            found_improvement = False
            # first time through
            if not self.all_pipelines:
                metric_val = CrossValidationStat()
                best_predictions = getattr(pipelines[0], which_result).predictions
                best_pipeline = pipelines[0]
                best_metrics = getattr(pipelines[0], which_result).metric_values
                best_score = np.mean(np.array([a for a in best_metrics.values()]))
                metric_val.add_fold_metric(metric, getattr(pipelines[0], which_result).metric_values) 
                #= getattr(pipelines[0], which_result).stat
                found_improvement = True
                print('Best single pipeline score ',  str(best_score))
            else:
                for pipeline in pipelines:
                    metric_val = CrossValidationStat()
                    metric_values = {}

                    if median:
                        y_temp = self._add_median_prediction(self.all_pipelines, getattr(pipeline, which_result))
                    else:
                        y_temp = self._add_mean_prediction(self.predictions, getattr(pipeline, which_result))

                    y_rounded = np.rint(y_temp) if self.discrete_metric else y_temp

                    kf = KFold(n_splits = cv, shuffle = True, random_state = seed)
                    for i in range(0, len(self.problem.metrics)):
                        metric = self.problem.metrics[i]
                        fn = self.problem.metric_functions[i]
                        for k, (train, test) in enumerate(kf.split(X, y)):
                            yfold = y.take(test, axis = 0).values.ravel()
                            yround_fold = np.take(y_rounded, test, axis =0).ravel()
                            fold_score = self._call_function(fn, yfold, yround_fold)
                            metric_val.add_fold_metric(metric, fold_score)
                            #metric_val = self._call_function(fn, y, y_rounded)
                        if fold_score is None:
                            return None
                        metric_values[metric.name] = metric_val.get_metric(metric) #metric_val

                    score_improve = [v - best_metrics[k] for k, v in metric_values.items()]
                    score_improve = [score_improve[l] * (-1 if self.minimize_metric[l] else 1) for l in range(len(score_improve))]
                    score_improve = np.mean(np.array([a for a in score_improve]))
                    score = np.mean(np.array([a for a in metric_values.values()]))

                    if (score_improve > 0):
                        best_score = score
                        best_pipeline = pipeline
                        print('Adding Pipeline ', pipeline)
                        best_predictions = pd.DataFrame(y_temp, index = X.index, columns = y.columns)
                        best_metrics = metric_values
                        best_stat = metric_val
                        found_improvement = True

            if found_improvement:
                self.all_pipelines.append(best_pipeline)
                self.predictions = best_predictions
                self.metric_values = best_metrics
                self.cv_stat = best_stat 
                self.score = best_score

        print('Found ensemble of size ', str(len(self.all_pipelines)), ' with score ',  str(self.score))
        ensemble_pipeline_ids = [pl.id for pl in self.all_pipelines]
        unique, indices, counts = np.unique(ensemble_pipeline_ids, return_index = True, return_counts = True)
        self.pipelines = [self.all_pipelines[index] for index in sorted(indices)]
        self.pipeline_weights = dict(zip(unique, counts))
        self.pipeline_weights = [self.pipeline_weights[p.id] for p in self.pipelines]

        if plan:
            self.train_result = PipelineExecutionResult(self.predictions, self.metric_values, self.cv_stat)
        else:
            self.test_result = PipelineExecutionResult(self.predictions, self.metric_values, self.cv_stat)


        ens_pipeline = self._ens_pipeline()
        print('Ensemble Pipeline ID ', ens_pipeline.id)
        return ens_pipeline

    def _ens_pipeline(self):
        ens_pl = Pipeline(ensemble = self)

        ens_pl.planner_result = self.train_result
        ens_pl.finished = True

        return ens_pl

    def _add_median_prediction(self, pipelines, new_result):
        all_preds = [p.values for p in self.all_pipelines]
        all_preds.append(new_result)
        all_preds = np.concatenate(all_preds, axis = -1) # join predictions as columns
        return np.median(all_preds, axis = -1)


    def _add_mean_prediction(self, old_prediction, new_result, old_weight = None, new_weight = None, median = True):
        # TODO : NON-NUMERIC LABELS
        if new_weight is None or old_weight is None:
            _divisor = (1.0*len(self.all_pipelines)+1)
            _multiplier = len(self.all_pipelines)
        else:
            _divisor = old_weight + new_weight
            _multiplier = old_weight

        y_temp = (old_prediction.values * _multiplier + new_result.predictions.values) / _divisor
        #temp_predictions = (self.predictions[self.predictions.select_dtypes(include=['number']).columns] * len(self.all_pipelines)
        #                   + pipeline.predictions) / (len(self.all_pipelines)+1)


        return y_temp

    def _call_function(self, scoring_function, *args):
        mod = inspect.getmodule(scoring_function)
        try:
            module = importlib.import_module(mod.__name__)
            return scoring_function(*args)
        except Exception as e:
            sys.stderr.write("ERROR _call_function %s: %s\n" % (scoring_function, e))
            traceback.print_exc()
            return None
