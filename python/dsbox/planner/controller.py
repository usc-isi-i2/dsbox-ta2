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

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.schema.data_profile import DataProfile
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.executer.executionhelper import ExecutionHelper
from dsbox.planner.common.pipeline import Pipeline, PipelineExecutionResult

class Feature:
    def __init__(self, data_directory, feature_id):
        self.data_directory = data_directory
        self.feature_id = feature_id

class Controller(object):
    """
    This is the overall "planning" coordinator. It is passed in the data directory
    and the primitives library directory, and it generates plans by calling out to L1, L2
    and L3 planners.
    """
    def __init__(self, libdir):
        self.libdir = os.path.abspath(libdir)
        self.columns = []
        self.targets = []
        self.train_data = None
        self.train_labels = None
        self.indexcol = None
        self.media_type = None
        self.exec_pipelines = []
        self.test_data = None
        self.test_indexcol = None
        self.l1_planner = None
        self.l2_planner = None

    '''
    Set config directories and data schema file
    '''
    def set_config(self, config):
        self.data_schema = config.get('dataset_schema', None)
        self.problem_schema = config.get('problem_schema', None)
        self.train_dir = self._dir(config, 'training_data_root')
        self.log_dir = self._dir(config, 'pipeline_logs_root')
        self.exec_dir = self._dir(config, 'executables_root')
        self.tmp_dir = self._dir(config, 'temp_storage_root')

        # Create some debugging files
        self.logfile = open("%s%slog.txt" % (self.tmp_dir, os.sep), 'w')
        self.errorfile = open("%s%sstderr.txt" % (self.tmp_dir, os.sep), 'w')
        self.pipelinesfile = open("%s%spipelines.txt" % (self.tmp_dir, os.sep), 'w')

        # Redirect stderr to error file
        sys.stderr = self.errorfile

    '''
    Set config directories and schema from just datadir and outputdir
    '''
    def set_config_simple(self, datadir, outputdir):
        self.set_config({
            'dataset_schema': datadir + os.sep + "dataSchema.json",
            'problem_schema': datadir + os.sep + ".." + os.sep + "problemSchema.json",
            'training_data_root': datadir,
            'pipeline_logs_root': outputdir + os.sep + "logs",
            'executables_root': outputdir + os.sep + "executables",
            'temp_storage_root': outputdir + os.sep + "temp"
        })

    """
    This function creates train_data and train_labels from trainData.csv and trainTargets.csv
    """
    def initialize_data_from_defaults(self):
        print("Loading Data..")

        self.helper = ExecutionHelper(self.train_dir, self.exec_dir, None, self.data_schema)
        self.columns = self.helper.columns
        self.targets = self.helper.targets
        self.indexcol = self.helper.indexcol
        self.media_type = self.helper.media_type
        self.train_data = self.helper.read_data(self.train_dir + os.sep + 'trainData.csv.gz',
                                                 self.columns, self.indexcol)
        self.train_labels = self.helper.read_data(self.train_dir + os.sep + 'trainTargets.csv.gz',
                                                 self.targets, self.indexcol, labeldata=True)
        self.helper.tmp_dir = self.tmp_dir

    """
    This function creates train_data and train_labels from the set of train and target features
    """
    def initialize_data_from_features(self, train_features, target_features):
        data_train_features_map = {}
        data_target_features_map = {}

        print("Loading Data..")

        for feature in train_features:
            data_train_features = data_train_features_map.get(feature.data_directory, [])
            data_train_features.append(feature.feature_id)
            data_train_features_map[feature.data_directory] = data_train_features

        for feature in target_features:
            data_target_features = data_target_features_map.get(feature.data_directory, [])
            data_target_features.append(feature.feature_id)
            data_target_features_map[feature.data_directory] = data_target_features

        for data_directory in data_train_features_map.keys():
            helper = ExecutionHelper(data_directory, self.exec_dir)
            # FIXME: Should handle multiple schema files here
            self.data_schema = helper.schema_file
            indexcol = helper.indexcol
            columns = []
            data_train_features = data_train_features_map[data_directory]
            for col in helper.columns:
                if (col['varName'] in data_train_features) or ("*" in data_train_features):
                    columns.append(col)
            train_data = helper.read_data(data_directory + os.sep + 'trainData.csv.gz',
                                         columns, indexcol)
            if self.train_data is None:
                self.train_data = train_data
                self.columns = columns
            else:
                self.train_data = pd.concat([self.train_data, train_data], axis=1)
                self.columns = self.columns + columns
                self.train_data.columns = self.columns

            self.helper = helper
            if indexcol is not None:
                self.indexcol = indexcol
            if helper.media_type is not None:
                self.media_type = helper.media_type

        for data_directory in data_target_features_map.keys():
            helper = ExecutionHelper(data_directory, self.exec_dir)
            indexcol = helper.indexcol
            targets = []
            data_target_features = data_target_features_map[data_directory]
            for col in helper.targets:
                if (col['varName'] in data_target_features) or ("*" in data_target_features):
                    targets.append(col)

            train_labels = helper.read_data(data_directory + os.sep + 'trainTargets.csv.gz',
                                         targets, indexcol)
            if self.train_labels is None:
                self.train_labels = train_labels
                self.targets = targets
            else:
                self.train_labels = pd.concat([self.train_labels, train_labels], axis=1)
                self.targets = self.targets + targets
                self.train_labels.columns = self.targets

        self.helper.columns = self.columns
        self.helper.targets = self.targets
        self.helper.indexcol = self.indexcol


    """
    This function creates test_data from the set of test features
    """
    def initialize_test_data_from_features(self, test_features):
        data_test_features_map = {}

        print("Loading Data..")
        self.test_data = None

        for feature in test_features:
            data_test_features = data_test_features_map.get(feature.data_directory, [])
            data_test_features.append(feature.feature_id)
            data_test_features_map[feature.data_directory] = data_test_features

        for data_directory in data_test_features_map.keys():
            helper = ExecutionHelper(data_directory, self.exec_dir)
            indexcol = helper.indexcol
            columns = []
            data_test_features = data_test_features_map[data_directory]
            for col in helper.columns:
                if (col['varName'] in data_test_features) or ("*" in data_test_features):
                    columns.append(col)
            test_data = helper.read_data(data_directory + os.sep + 'testData.csv.gz',
                                         columns, indexcol)
            if self.test_data is None:
                self.test_data = test_data
                self.test_columns = columns
            else:
                self.test_data = pd.concat([self.test_data, test_data], axis=1)
                self.test_columns = self.test_columns + columns

            self.test_helper = helper
            if indexcol is not None:
                self.test_indexcol = indexcol
            if helper.media_type is not None:
                self.test_media_type = helper.media_type

        self.test_helper.columns = self.test_columns
        self.test_helper.indexcol = self.test_indexcol

    """
    Set the task type, metric and output type via the schema
    """
    def load_problem_schema(self):
        self.problem = self.helper.load_json(self.problem_schema)
        self.helper.set_task_type(
            self.problem.get('taskType', None),
            self.problem.get('taskSubType', None)
        )
        self.helper.set_metric(self.problem.get('metric', 'accuracy'))
        self.helper.problemid = self.problem['problemId']
        self.set_output_type(None)

    """
    Set the output type
    """
    def set_output_type(self, output_type):
        self.output_type = output_type

    """
    Initialize the L1 and L2 planners
    """
    def initialize_planners(self):
        self.l1_planner = LevelOnePlannerProxy(self.libdir, self.helper)
        self.l2_planner = LevelTwoPlanner(self.libdir, self.helper)

    """
    Train and select pipelines
    """
    def train(self, planner_event_handler, cutoff=10):
        self.logfile.write("Task type: %s\n" % self.helper.task_type)
        self.logfile.write("Metrics: %s\n" % self.helper.metrics)

        pe = planner_event_handler

        self._show_status("Planning...")

        # Get data details
        df = pd.DataFrame(self.train_data, columns = self.train_data.columns)
        df_lbl = pd.DataFrame(self.train_labels, columns = self.train_labels.columns)

        df_profile = self._get_data_profile(df)
        self.logfile.write("Data profile: %s\n" % df_profile)

        l1_pipelines_handled = {}
        l2_pipelines_handled = {}
        l1_pipelines = self.l1_planner.get_pipelines(df)
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
                            yield pe.SubmittedPipeline(l2_pipeline)

            self.logfile.write("\nL2 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_pipelines))

            self._show_status("Found %d executable pipeline(s). Testing them..." % len(l2_pipelines))

            for l2_pipeline in l2_pipelines:
                yield pe.RunningPipeline(l2_pipeline)

                # TODO: Execute in parallel (fork, or separate thread)
                exec_pipeline = self.l2_planner.patch_and_execute_pipeline(
                        l2_pipeline, df, df_lbl, self.columns)
                l2_pipelines_handled[str(l2_pipeline)] = True
                yield pe.CompletedPipeline(l2_pipeline, exec_pipeline)

                if exec_pipeline:
                    self.exec_pipelines.append(exec_pipeline)

            self.exec_pipelines = sorted(self.exec_pipelines, key=lambda x: self._sort_by_metric(x))
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

        # Ended planners
        self._show_status("Found total %d successfully executing pipeline(s)..." % len(self.exec_pipelines))

        # Copy over the data schema
        data_schema_file = str(uuid.uuid4()) + ".json"
        shutil.copyfile(self.helper.schema_file, self.tmp_dir + os.sep + data_schema_file)

        # Create executables
        self.pipelinesfile.write("# Pipelines ranked by metrics (%s)\n" % self.helper.metrics)
        for index in range(0, len(self.exec_pipelines)):
            pipeline = self.exec_pipelines[index]
            rank = index + 1
            # Format the metric values
            metric_values = []
            for metric in pipeline.planner_result.metric_values.keys():
                metric_value = pipeline.planner_result.metric_values[metric]
                metric_values.append("%s = %2.4f" % (metric, metric_value))

            self.pipelinesfile.write("%s ( %s ) : %s\n" % (pipeline.id, pipeline, metric_values))
            self.helper.create_pipeline_executable(pipeline, data_schema_file)
            self.create_pipeline_logfile(pipeline, rank)

    '''
    Predict results on test data given a pipeline
    '''
    def test(self, pipeline, test_event_handler):
        helper = self.test_helper
        testdf = self.test_data
        print("** Evaluating pipeline %s" % str(pipeline))
        for primitive in pipeline.primitives:
            # Initialize primitive
            try:
                print("Executing %s" % primitive)
                if primitive.task == "Modeling":
                    result = pd.DataFrame(primitive.executables.predict(testdf), index=testdf.index, columns=[self.targets[1]['varName']])
                    pipeline.test_result = PipelineExecutionResult(result, None)
                    break
                elif primitive.task == "PreProcessing":
                    testdf = helper.test_execute_primitive(primitive, testdf)
                elif primitive.task == "FeatureExtraction":
                    testdf = helper.test_featurise(testdf, primitive)
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
            "problem_id": self.helper.problemid,
            "pipeline_rank": rank,
            "name": pipeline.id,
            "primitives": []
        }
        for primitive in pipeline.primitives:
            logdata['primitives'].append(primitive.cls)
        with(open(logfilename, 'w')) as pipelog:
            json.dump(logdata, pipelog,
                sort_keys=True, indent=4, separators=(',', ': '))
            pipelog.close()

    def _dir(self, config, key):
        dir = config.get(key)
        if dir is None:
            return None
        dir = os.path.abspath(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def _show_status(self, status):
        sys.stdout.write("%s\n" % status)
        sys.stdout.flush()

    def _sort_by_metric(self, pipeline):
        # NOTE: Sorting/Ranking by first metric only
        metric_name = self.helper.metrics[0].name
        mlower = metric_name.lower()
        if "error" in mlower or "loss" in mlower or "time" in mlower:
            return pipeline.planner_result.metric_values[metric_name]
        return -pipeline.planner_result.metric_values[metric_name]

    def _get_data_profile(self, df):
        return DataProfile(df)
