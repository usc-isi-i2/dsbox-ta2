import os
import sys
import os.path
import uuid
import copy
import math
import json
import numpy
import shutil
import traceback
import pandas as pd

from typing import List

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.schema.data_profile import DataProfile
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.executer.executionhelper import ExecutionHelper
from dsbox.planner.common.data_manager import Dataset, DataManager
from dsbox.planner.common.pipeline import Pipeline, PipelineExecutionResult, OneStandardErrorPipelineSorter, PipelineSorter
from dsbox.planner.common.problem_manager import Problem
from dsbox.planner.common.resource_manager import ResourceManager
from dsbox.planner.ensemble import Ensemble

class Feature:
    def __init__(self, resource_id, feature_name):
        self.resource_id = resource_id
        self.feature_name = feature_name

class Controller(object):
    problem = None
    dataset = None
    execution_helper = None
    resource_manager = None

    config = None
    num_cpus = 0
    ram = 0
    timeout = 60
    include_primitives = []
    include_families = []
    exclude_primitives = []
    exclude_families = []
    #max_ensemble = 5

    exec_pipelines = []
    l1_planner = None
    l2_planner = None

    """
    This is the overall "planning" coordinator. It is passed in the data directory
    and the primitives library directory, and it generates plans by calling out to L1, L2
    and L3 planners.
    """
    def __init__(self, libdir):
        # FIXME: This should change to the primitive discovery interface
        self.libdir = os.path.abspath(libdir)

        self.exec_pipelines: List[Pipeline] = []
        self._pipeline_sorter: PipelineSorter = None

    '''
    Set config directories and data schema file
    '''
    def initialize_from_config(self, config):
        self.config = config

        self.log_dir = self._dir(config, 'pipeline_logs_root', True)
        self.exec_dir = self._dir(config, 'executables_root', True)
        self.tmp_dir = self._dir(config, 'temp_storage_root', True)

        self.num_cpus = int(config.get('cpus', 0))
        self.ram = config.get('ram', 0)
        self.timeout = (config.get('timeout', 60))*60
        #self.max_ensemble = int(config.get('max_ensemble', 0))

        # Create some debugging files
        self.logfile = open("%s%slog.txt" % (self.tmp_dir, os.sep), 'w')
        self.errorfile = open("%s%sstderr.txt" % (self.tmp_dir, os.sep), 'w')
        self.pipelinesfile = open("%s%spipelines.txt" % (self.tmp_dir, os.sep), 'w')

        self.problem = Problem()
        self.data_manager = DataManager()
        self.execution_helper = ExecutionHelper(self.problem, self.data_manager)
        self.resource_manager = ResourceManager(self.execution_helper, self.num_cpus)

        self.include_primitives = config.get('include_primitives', [])
        self.include_primitives.extend(config.get('include_families', []))
        self.exclude_primitives = config.get('exclude_primitives', [])
        self.exclude_primitives = config.get('exclude_families', [])
        # Redirect stderr to error file
        sys.stderr = self.errorfile

    '''
    Set config directories and schema from just problemdir, datadir and outputdir
    '''
    def initialize_simple(self, problemdir, datadir, outputdir):
        self.initialize_from_config(
            self.create_simple_config(problemdir, datadir, outputdir)
        )

    '''
    Create config from problemdir, datadir, outputdir
    '''
    def create_simple_config(self, problemdir, datadir, outputdir):
        return {
            "problem_root": problemdir,
            "problem_schema": problemdir + os.sep + 'problemDoc.json',
            "training_data_root": datadir,
            "dataset_schema": datadir + os.sep + 'datasetDoc.json',
            'pipeline_logs_root': outputdir + os.sep + "logs",
            'executables_root': outputdir + os.sep + "executables",
            'temp_storage_root': outputdir + os.sep + "temp",
            "timeout": 60,
            "cpus"  : "4",
            "ram"   : "4Gi"
            #"max_ensemble" : 5
            }


    """
    Set the task type, metric and output type via the schema
    """
    def load_problem(self):
        problemroot = self._dir(self.config, 'problem_root')
        problemdoc = self.config.get('problem_schema', None)
        assert(problemroot is not None)
        self.problem.load_problem(problemroot, problemdoc)

    """
    Initialize data from the config
    """
    def initialize_training_data_from_config(self):
        dataroot = self._dir(self.config, 'training_data_root')
        datadoc = self.config.get('dataset_schema', None)
        assert(dataroot is not None)
        dataset = Dataset()
        dataset.load_dataset(dataroot, datadoc)
        self.data_manager.initialize_data(self.problem, [dataset], view='TRAIN')

    """
    Initialize from features

    - Used by TA3
    """
    def initialize_from_features_simple(self, datafile, train_features, target_features, outputdir, view=None):
        data_directory = os.path.dirname(datafile)
        config = self.create_simple_config(outputdir, data_directory, outputdir)
        self.initialize_from_features(datafile, train_features, target_features, config, view)

    """

    Initialize all from features and config
    - Used by TA3
    """

    def initialize_from_features(self, datafile, train_features, target_features, config, view=None):
        self.initialize_from_config(config)
        data_directory = os.path.dirname(datafile)

        # Load datasets first
        filters = {}
        targets = {}
        dataset = Dataset()
        dataset.load_dataset(data_directory, datafile)

        if train_features is not None:
            filters[dataset.dsID] = list(map(
                lambda x: {"resID": x.resource_id, "colName": x.feature_name}, train_features
            ))
            self.problem.dataset_filters = filters

        if target_features is not None:
            targets[dataset.dsID] = list(map(
                lambda x: {"resID": x.resource_id, "colName": x.feature_name}, target_features
            ))
            self.problem.dataset_targets = targets

        self.data_manager.initialize_data(self.problem, [dataset], view)

    def get_pipeline_sorter(self):
        if self._pipeline_sorter is None:
            self._pipeline_sorter = OneStandardErrorPipelineSorter(self.problem.metrics[0])
        return self._pipeline_sorter


    """
    Initialize the L1 and L2 planners
    """
    def initialize_planners(self):
        self.l1_planner = LevelOnePlannerProxy(self.libdir, self.execution_helper, include = self.include_primitives, exclude = self.exclude_primitives)
        self.l2_planner = LevelTwoPlanner(self.libdir, self.execution_helper)

    """
    Train and select pipelines
    """
    def train(self, planner_event_handler, cutoff=10, ensemble = True):
        self.exec_pipelines = []
        self.l2_planner.primitive_cache = {}
        self.l2_planner.execution_cache = {}

        self.logfile.write("Task type: %s\n" % self.problem.task_type)
        self.logfile.write("Metrics: %s\n" % self.problem.metrics)

        if ensemble:
            self.ensemble = Ensemble(self.problem) #,self.max_ensemble)

        pe = planner_event_handler

        self._show_status("Planning...")

        # Get data details
        df = copy.copy(self.data_manager.input_data)
        df_lbl = copy.copy(self.data_manager.target_data)
        df_profile = DataProfile(df)
        self.logfile.write("Data profile: %s\n" % df_profile)
        l2_pipelines_map = {}
        l1_pipelines_handled = {}
        l2_pipelines_handled = {}
        l1_pipelines = self.l1_planner.get_pipelines(df)
        if l1_pipelines is None:
            # If no L1 Pipelines, then we don't support this problem
            yield pe.ProblemNotImplemented()
            return
        self.exec_pipelines = []

        while len(l1_pipelines) > 0:
            self.logfile.write("\nL1 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l1_pipelines))
            self.logfile.write("-------------\n")

            l2_l1_map = {}

            self._show_status("Exploring %d basic pipeline(s)..." % len(l1_pipelines))

            l2_pipelines = []
            for l1_pipeline in l1_pipelines:
                if l1_pipelines_handled.get(str(l1_pipeline), False):
                    continue
                l2_pipeline_list = self.l2_planner.expand_pipeline(l1_pipeline, df_profile)
                l1_pipelines_handled[str(l1_pipeline)] = True
                if l2_pipeline_list:
                    for l2_pipeline in l2_pipeline_list:
                        if not l2_pipelines_handled.get(str(l2_pipeline), False):
                            l2_l1_map[l2_pipeline.id] = l1_pipeline
                            l2_pipelines.append(l2_pipeline)
                            l2_pipelines_map[str(l2_pipeline)] = l2_pipeline
                            yield pe.SubmittedPipeline(l2_pipeline)

            self.logfile.write("\nL2 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_pipelines))

            self._show_status("Found %d executable pipeline(s). Testing them..." % len(l2_pipelines))

            for l2_pipeline in l2_pipelines:
                yield pe.RunningPipeline(l2_pipeline)
                # exec_pipeline = self.l2_planner.patch_and_execute_pipeline(l2_pipeline, df, df_lbl)

            exec_pipelines = self.resource_manager.execute_pipelines(l2_pipelines, df, df_lbl)
            for exec_pipeline in exec_pipelines:
                l2_pipeline = l2_pipelines_map[str(exec_pipeline)]
                l2_pipelines_handled[str(l2_pipeline)] = True
                yield pe.CompletedPipeline(l2_pipeline, exec_pipeline)
                if exec_pipeline:
                    self.exec_pipelines.append(exec_pipeline)

            # self.exec_pipelines = sorted(self.exec_pipelines, key=lambda x: self._sort_by_metric(x))
            self.exec_pipelines = self.get_pipeline_sorter().sort_pipelines(self.exec_pipelines)
            self.logfile.write("\nL2 Executed Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(self.exec_pipelines))

            # TODO: Do Pipeline Hyperparameter Tuning

            # Pick top N pipelines, and get similar pipelines to it from the L1 planner to further explore
            l1_related_pipelines = []
            for index in range(0, cutoff):
                if index >= len(self.exec_pipelines):
                    break
                l1_pipeline = l2_l1_map.get(self.exec_pipelines[index].id)
                if l1_pipeline:
                    related_pipelines = self.l1_planner.get_related_pipelines(l1_pipeline)
                    for related_pipeline in related_pipelines:
                        if not l1_pipelines_handled.get(str(related_pipeline), False):
                            l1_related_pipelines.append(related_pipeline)

            self.logfile.write("\nRelated L1 Pipelines to top %d L2 Pipelines:\n-------------\n" % cutoff)
            self.logfile.write("%s\n" % str(l1_related_pipelines))
            l1_pipelines = l1_related_pipelines

        if ensemble:
            try:
                ensemble_pipeline = self.ensemble.greedy_add(self.exec_pipelines, df, df_lbl)
                if ensemble_pipeline:
                    self.exec_pipelines.append(ensemble_pipeline)
            except Exception as e:
                traceback.print_exc()
                sys.stderr.write("ERROR ensemble.greedy_add : %s\n" % e)

        self.write_training_results()

    '''
    Write training results to file
    '''
    def write_training_results(self):
        # Sort pipelines
        # self.exec_pipelines = sorted(self.exec_pipelines, key=lambda x: self._sort_by_metric(x))
        self.exec_pipelines = self.get_pipeline_sorter().sort_pipelines(self.exec_pipelines)


        # Ended planners
        self._show_status("Found total %d successfully executing pipeline(s)..." % len(self.exec_pipelines))

        # Create executables
        self.pipelinesfile.write("# Pipelines ranked by (adjusted) metrics (%s)\n" % self.problem.metrics)
        for index in range(0, len(self.exec_pipelines)):
            pipeline = self.exec_pipelines[index]
            rank = index + 1
            # Format the metric values
            metric_values = []
            for metric in pipeline.planner_result.metric_values.keys():
                metric_value = pipeline.planner_result.metric_values[metric]
                metric_values.append("%s = %2.4f" % (metric, metric_value))

            self.pipelinesfile.write("%s ( %s ) : %s\n" % (pipeline.id, pipeline, metric_values))
            self.execution_helper.create_pipeline_executable(pipeline, self.config)
            self.create_pipeline_logfile(pipeline, rank)

    '''
    Predict results on test data given a pipeline
    '''
    def test(self, pipeline, test_event_handler):
        helper = ExecutionHelper(self.problem, self.data_manager)
        testdf = pd.DataFrame(copy.copy(self.data_manager.input_data))
        target_col = self.data_manager.target_columns[0]['colName']
        print("** Evaluating pipeline %s" % str(pipeline))
        sys.stdout.flush()
        for primitive in pipeline.primitives:
            # Initialize primitive
            try:
                print("Executing %s" % primitive)
                sys.stdout.flush()
                if primitive.task == "Modeling":
                    if primitive.unified_interface:
                        result = pd.DataFrame(primitive.executables.produce(inputs=testdf).value, index=testdf.index, columns=[target_col])
                    else:
                        result = pd.DataFrame(primitive.executables.predict(testdf), index=testdf.index, columns=[target_col])
                    pipeline.test_result = PipelineExecutionResult(result, None)
                    break
                elif primitive.task == "PreProcessing":
                    testdf = helper.test_execute_primitive(primitive, testdf)
                elif primitive.task == "FeatureExtraction":
                    testdf = helper.test_featurise(primitive, testdf)
                if testdf is None:
                    break
            except Exception as e:
                sys.stderr.write(
                    "ERROR test(%s) : %s\n" % (pipeline, e))
                traceback.print_exc()

        yield test_event_handler.ExecutedPipeline(pipeline)

    def stop(self):
        '''
        Stop planning, and write out the current list (sorted by metric)
        '''

    def create_pipeline_logfile(self, pipeline, rank):
        logfilename = "%s%s%s.json" % (self.log_dir, os.sep, pipeline.id)
        logdata = {
            "problem_id": self.problem.prID,
            "pipeline_rank": rank,
            "name": pipeline.id,
            "primitives": []
        }
        primitive_set = set()
        ensembling = pipeline.ensemble is not None
        if ensembling:
            for epipe in pipeline.ensemble.pipelines:
                for primitive in epipe.primitives:
                    primitive_set.add(primitive.cls)
        else:
            for primitive in pipeline.primitives:
                primitive_set.add(primitive.cls)

        logdata["primitives"] = list(primitive_set)
        with(open(logfilename, 'w')) as pipelog:
            json.dump(logdata, pipelog,
                sort_keys=True, indent=4, separators=(',', ': '))
            pipelog.close()

    def _dir(self, config, key, makeflag=False):
        dir = config.get(key)
        if dir is None:
            return None
        dir = os.path.abspath(dir)
        if makeflag and not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def _show_status(self, status):
        print(status)
        sys.stdout.flush()

    def _sort_by_metric(self, pipeline):
        # NOTE: Sorting/Ranking by first metric only
        if self.problem.metrics[0].larger_is_better():
            return -pipeline.planner_result.metric_values[self.problem.metrics[0].name]
        return pipeline.planner_result.metric_values[self.problem.metrics[0].name]
