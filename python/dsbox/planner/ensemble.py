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

MIN_METRICS = [Metric.MEAN_SQUARED_ERROR, Metric.ROOT_MEAN_SQUARED_ERROR, Metric.ROOT_MEAN_SQUARED_ERROR_AVG, Metric.MEAN_ABSOLUTE_ERROR, Metric.EXECUTION_TIME]
DISCRETE_METRIC = [TaskSubType.BINARY, TaskSubType.MULTICLASS, TaskSubType.MULTILABEL, TaskSubType.OVERLAPPING, TaskSubType.NONOVERLAPPING]

class Ensemble(object):
    def __init__(self, problem, max_pipelines = 5):
        self.max_pipelines = max_pipelines
        self.predictions = None  # Predictions dataframe
        self.metric_values =  None # Dictionary of metric to value
        self.train_result = None
        self.test_result = None
        self.score = 0
        self.all_pipelines = []
        #self.test_pipeline_ids = []
        self.problem = problem
        self._analyze_metrics()

    def _analyze_metrics(self):
        # *** ONLY CONSIDERS 1 METRIC ***
        #self.minimize_metric = True if self.problem.metrics[0] in MIN_METRICS else False
        self.minimize_metric = []
        for i in range(0, len(self.problem.metrics)):
            print(self.problem.metrics[i])
            self.minimize_metric.append(True if self.problem.metrics[i] in MIN_METRICS else False)
        self.discrete_metric = True if self.problem.task_subtype in DISCRETE_METRIC else False


    def greedy_add(self, pipelines, X, y, max_pipelines = None):
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
                best_predictions = pipelines[0].planner_result.predictions
                best_pipeline = pipelines[0]
                best_metrics = pipelines[0].planner_result.metric_values
                best_score = np.mean(np.array([a for a in best_metrics.values()]))
                found_improvement = True
                print('Best single pipeline score ',  str(best_score))
            else:
                for pipeline in pipelines:
                    metric_values = {}
                      
                    y_temp = self._add_new_prediction(self.predictions, pipeline.planner_result)
                    
                    y_rounded = np.rint(y_temp) if self.discrete_metric else y_temp

                    for i in range(0, len(self.problem.metrics)):
                        metric = self.problem.metrics[i]
                        fn = self.problem.metric_functions[i]
                        metric_val = self._call_function(fn, y, y_rounded)
                        if metric_val is None:
                            return None
                        metric_values[metric.name] = metric_val
                    
                    score_improve = [v - best_metrics[k] for k, v in metric_values.items()]
                    score_improve = [score_improve[l] * (-1 if self.minimize_metric[l] else 1) for l in range(len(score_improve))]
                    score_improve = np.mean(np.array([a for a in score_improve]))
                    score = np.mean(np.array([a for a in metric_values.values()]))
                    
                    #print('Evaluating ', pipeline.primitives, score, score_improve)
                    if (score_improve > 0):
                        best_score = score
                        best_pipeline = pipeline
                        best_predictions = pd.DataFrame(y_temp, index = X.index, columns = y.columns)
                        best_metrics = metric_values
                        found_improvement = True
                    
            if found_improvement:
                self.all_pipelines.append(best_pipeline)
                self.predictions = best_predictions
                self.metric_values = best_metrics
                self.score = best_score                
        
        print('Found ensemble of size ', str(len(self.all_pipelines)), ' with score ',  str(self.score))        
        
        ensemble_pipeline_ids = [pl.id for pl in self.all_pipelines]
        unique, indices, counts = np.unique(ensemble_pipeline_ids, return_index = True, return_counts = True) 
        self.pipelines = [self.all_pipelines[index] for index in sorted(indices)]
        self.pipeline_weights = dict(zip(unique, counts))
        self.pipeline_weights = [self.pipeline_weights[p.id] for p in self.pipelines]
        self.train_result = PipelineExecutionResult(self.predictions, self.metric_values)

        ens_pipeline = self._ens_pipeline()
        print(ens_pipeline.id)
        return ens_pipeline

    def _ens_pipeline(self):
        ens_pl = Pipeline(ensemble = self)
        
        ens_pl.planner_result = self.train_result
        ens_pl.finished = True
        
        #print('**** ENSEMBLE PIPELINE ID ****', ens_pl.id)
        #try:   
            # if execution time too high
        #    ens_pl.start_time = sum([pl.start_time for pl in self.all_pipelines])
        #    ens_pl.end_time = sum([pl.end_time for pl in self.all_pipelines])
        #    print ('Ensemble Pipeline Execution Time: ', ens_pl.end_time - ens_pl.start_time)
        #except:
        #    pass

        
        #ens_pl.ensemble_pipelines = self.pipelines
        #ens_pl.ensemble_weights = self.pipeline_weights
        #ens_pl.round_predictions = self.discrete_metric

        return ens_pl

        #self.full_test_ensemble()
        #self.intermediate_test_ensemble()

    def _add_new_prediction(self, old_prediction, new_result, old_weight = None, new_weight = None):
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


    # NOT CURRENTLY USED
    def full_test_ensemble(self):
        # sequential add
        for i in range(len(self.all_pipelines)):
            if self.all_pipelines[i].test_result is not None:
                self.test_result.predictions = self._add_new_prediction(self.test_result.predictions, self.all_pipelines[i].test_result)
            else:
                print('No test result for pipeline: ', self.all_pipelines[i])
        if self.discrete_metric and self.test_result is not None:
            self.test_result.predictions.values = np.rint(self.test_result.predictions.values)


    def intermediate_test_ensemble(self):
        for i in range(len(self.pipelines)):
            # inefficient? goes through all pipelines each time
            if self.pipelines[i].test_result is not None:
                new_weight = self.pipeline_weights[i]
                if i == 0: #self.test_result is None: #i ==0:
                    self.test_result = PipelineExecutionResult(self.pipelines[i].test_result.predictions, 
                                                                self.pipelines[i].test_result.metric_values)
                    ensemble_weight = new_weight
                else:
                    #if self.all_pipelines[i].id not in self.test_pipelines_ids:
                    self.test_result.predictions = self._add_new_prediction(self.test_result.predictions, self.pipelines[i].test_result, 
                                ensemble_weight, ensemble_weight + new_weight)
                    ensemble_weight = ensemble_weight + new_weight
            else:
                print('No test result for pipeline: ', self.pipelines[i])
         
        if self.discrete_metric and self.test_result is not None:
            self.test_result.predictions.values = np.rint(self.test_result.predictions.values)
        

    
    #def _add_best_pipeline(self):
        # TRYING TO ADD BEST PIPELINE: sorting is backwards if minimization metric

                # y_temp = (self.predictions.values * len(self.all_pipelines) + pipeline.planner_result.predictions.values) / (1.0*len(self.all_pipelines)+1)
                # if self.discrete_metric:
                #         y_rounded = np.rint(y_temp)
                # else:
                #         y_rounded = y_temp
                # metric_values = {}
                # for i in range(0, len(self.problem.metrics)):
                #     metric = self.problem.metrics[i]
                #     fn = self.problem.metric_functions[i]
                #     metric_val = self._call_function(fn, y, y_rounded)
                #     if metric_val is None:
                #         return None
                #     metric_values[metric.name] = metric_val
                
                
                # self.all_pipelines.append(pipelines[0] if )
                # self.predictions = pd.DataFrame(y_temp, index = X.index, columns = y.columns)
                # best_metrics = metric_values
                # self.metric_values = metric_values

                # print('Adding BEST metric.  Did NOT find improvement.  Score : ', np.mean(np.array([a for a in metric_values.values()])))
