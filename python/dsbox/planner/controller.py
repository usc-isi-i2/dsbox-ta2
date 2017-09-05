import os
import sys
import os.path
import uuid
import math
import shutil
import pandas as pd

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.schema.data_profile import DataProfile
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.executer.executionhelper import ExecutionHelper

import sklearn.metrics
from sklearn.externals import joblib

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
    def __init__(self, train_features, target_features, libdir, outputdir):
        self.columns = []
        self.targets = []
        self.train_data = None
        self.train_labels = None
        self.indexcol = None
        self.media_type = None

        self.libdir = os.path.abspath(libdir)
        self.outputdir = os.path.abspath(outputdir)
        if os.path.exists(self.outputdir):
            shutil.rmtree(self.outputdir)
        os.makedirs(self.outputdir + os.sep + "models")
        os.makedirs(self.outputdir + os.sep + "executables")

        self.logfile = open("%s%slog.txt" % (self.outputdir, os.sep), 'w')
        self.errorfile = open("%s%sstderr.txt" % (self.outputdir, os.sep), 'w')
        self.pipelinesfile = open("%s%spipelines.txt" % (self.outputdir, os.sep), 'w')
        # Redirect stderr to error file
        sys.stderr = self.errorfile

        # Read training data
        self.read_feature_data(train_features, target_features)


    def read_feature_data(self, train_features, target_features):
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
            helper = ExecutionHelper(data_directory, self.outputdir)
            indexcol = helper.indexcol
            columns = []
            data_train_features = data_train_features_map[data_directory]
            for col in helper.columns:
                if (col['varName'] in data_train_features) or ("*" in data_train_features):
                    if(col['varRole'] != 'index'):
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
            helper = ExecutionHelper(data_directory, self.outputdir)
            indexcol = helper.indexcol
            targets = []
            data_target_features = data_target_features_map[data_directory]
            for col in helper.targets:
                if (col['varName'] in data_target_features) or ("*" in data_target_features):
                    if(col['varRole'] != 'index'):
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
    This function loads the problem details via the problem schema file
    """
    def load_problem_schema(self, jsonfile):
        self.problem = self.helper.load_json(jsonfile)
        self.set_task_type(
            self.problem.get('taskType', None),
            self.problem.get('taskSubType', None)
        )
        self.set_metric(self.problem.get('metric', 'accuracy'))
        self.set_output_type(None)

    def set_task_type(self, task_type, task_subtype=None):
        self.task_type = TaskType(task_type)
        self.task_subtype = None
        if task_subtype is not None:
            task_subtype = task_subtype.replace(self.task_type.value.title(), "")
            self.task_subtype = TaskSubType(task_subtype)

    def set_metric(self, metric):
        metric = metric[0].lower() + metric[1:]
        self.metric = Metric(metric)
        self.metric_function = self._get_metric_function(self.metric)

    def set_output_type(self, output_type):
        self.output_type = output_type

    def initialize_planners(self):
        self.l1_planner = LevelOnePlannerProxy(self.libdir, self.task_type, self.task_subtype, self.media_type)
        self.l2_planner = LevelTwoPlanner(self.libdir, self.helper)

    def convert_l1_to_l2(self, pipeline):
        pipeline.get_primitives()

    def show_status(self, status):
        sys.stdout.write("%s\n" % status)
        sys.stdout.flush()

    def start(self, planner_event_handler, cutoff=10):
        self.logfile.write("Task type: %s\n" % self.task_type)
        self.logfile.write("Metric: %s\n" % self.metric)

        pe = planner_event_handler

        self.show_status("Planning...")

        # Get data details
        df = pd.DataFrame(self.train_data, columns = self.train_data.columns)
        df_lbl = pd.DataFrame(self.train_labels, columns = self.train_labels.columns)

        df_profile = self._get_data_profile(df)
        self.logfile.write("Data profile: %s\n" % df_profile)

        l1_pipelines_handled = {}
        l2_pipelines_handled = {}
        l1_pipelines = self.l1_planner.get_pipelines(df)
        l2_exec_pipelines = []

        while len(l1_pipelines) > 0:
            self.logfile.write("\nL1 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l1_pipelines))
            self.logfile.write("-------------\n")

            l2_l1_map = {}

            self.show_status("Exploring %d basic pipeline(s)..." % len(l1_pipelines))

            l2_pipelines = []
            for l1_pipeline in l1_pipelines:
                if l1_pipelines_handled.get(str(l1_pipeline), False):
                    continue
                l2_pipeline_list = self.l2_planner.expand_pipeline(l1_pipeline, df_profile)
                l1_pipelines_handled[str(l1_pipeline)] = True
                if l2_pipeline_list:
                    for l2_pipeline in l2_pipeline_list:
                        if not l2_pipelines_handled.get(str(l2_pipeline), False):
                            l2_l1_map[str(l2_pipeline)] = l1_pipeline
                            l2_pipelines.append(l2_pipeline)
                            yield pe.SubmittedPipeline(l2_pipeline)

            self.logfile.write("\nL2 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_pipelines))

            self.show_status("Found %d executable pipeline(s). Testing them..." % len(l2_pipelines))

            for l2_pipeline in l2_pipelines:
                yield pe.RunningPipeline(l2_pipeline)

                # TODO: Execute parallelly (fork, or separate thread)
                expipe = self.l2_planner.patch_and_execute_pipeline(
                        l2_pipeline, df, df_lbl, self.columns, self.task_type, self.metric, self.metric_function)
                l2_pipelines_handled[str(l2_pipeline)] = True

                yield pe.CompletedPipeline(l2_pipeline, expipe)

                if expipe:
                    l2_exec_pipelines.append(expipe)

            l2_exec_pipelines = sorted(l2_exec_pipelines, key=lambda x: self._sort_by_metric(x))

            self.logfile.write("\nL2 Executed Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_exec_pipelines))

            # TODO: Do Pipeline Hyperparameter Tuning

            # Pick top N pipelines, and get similar pipelines to it from the L1 planner to further explore
            l1_related_pipelines = []
            for index in range(0, cutoff):
                if index >= len(l2_exec_pipelines):
                    break
                pipeline = l2_l1_map.get(str(l2_exec_pipelines[index][0]))
                if pipeline:
                    related_pipelines = self.l1_planner.get_related_pipelines(pipeline)
                    for related_pipeline in related_pipelines:
                        if not l1_pipelines_handled.get(str(related_pipeline), False):
                            l1_related_pipelines.append(related_pipeline)


            self.logfile.write("\nRelated L1 Pipelines to top %d L2 Pipelines:\n-------------\n" % cutoff)
            self.logfile.write("%s\n" % str(l1_related_pipelines))
            l1_pipelines = l1_related_pipelines

        # Ended planners
        self.show_status("Found total %d successfully executing pipeline(s)..." % len(l2_exec_pipelines))

        # Create executables
        self.pipelinesfile.write("# Pipelines ranked by metric (%s)\n" % self.metric)
        for index in range(0, len(l2_exec_pipelines)):
            pipeline = l2_exec_pipelines[index][0]
            self.pipelinesfile.write("%s : %2.4f\n" % (pipeline, l2_exec_pipelines[index][1]))
            pipeline_name = str(index+1) + "." + str(uuid.uuid1())
            self.helper.create_pipeline_executable(pipeline, pipeline_name)


    def _sort_by_metric(self, pipeline):
        if "error" in self.metric.value.lower():
            return pipeline[1]
        return -pipeline[1]

    def _get_metric_function(self, metric):
        if metric==Metric.ACCURACY:
            return sklearn.metrics.accuracy_score
        elif metric==Metric.F1:
            return sklearn.metrics.f1_score
        elif metric==Metric.F1_MICRO:
            return self.f1_micro
        elif metric==Metric.F1_MACRO:
            return self.f1_macro
        elif metric==Metric.ROC_AUC:
            return sklearn.metrics.roc_auc_score
        elif metric==Metric.ROC_AUC_MICRO:
            return self.roc_auc_micro
        elif metric==Metric.ROC_AUC_MACRO:
            return self.roc_auc_macro
        elif metric==Metric.ROOT_MEAN_SQUARED_ERROR:
            return self.root_mean_squared_error
        elif metric==Metric.ROOT_MEAN_SQUARED_ERROR_AVG:
            return self.root_mean_squared_error
        elif metric==Metric.MEAN_ABSOLUTE_ERROR:
            return sklearn.metrics.mean_absolute_error
        elif metric==Metric.R_SQUARED:
            return sklearn.metrics.r2_score
        elif metric==Metric.NORMALIZED_MUTUAL_INFORMATION:
            return sklearn.metrics.normalized_mutual_info_score
        elif metric==Metric.JACCARD_SIMILARITY_SCORE:
            return sklearn.metrics.jaccard_similarity_score
        return sklearn.metrics.accuracy_score

    def _get_data_profile(self, df):
        return DataProfile(df)

    def stop(self):
        '''
        Stop planning, and write out the current list (sorted by metric)
        '''

    ''' Custom Metric Functions '''
    def f1_micro(self, y_true, y_pred):
        return sklearn.metrics.f1_score(y_true, y_pred, average="micro")

    def f1_macro(self, y_true, y_pred):
        return sklearn.metrics.f1_score(y_true, y_pred, average="macro")

    def roc_auc_micro(self, y_true, y_pred):
        return sklearn.metrics.roc_auc_score(y_true, y_pred, average="micro")

    def roc_auc_macro(self, y_true, y_pred):
        return sklearn.metrics.roc_auc_score(y_true, y_pred, average="macro")

    def root_mean_squared_error(self, y_true, y_pred):
        import math
        return math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
