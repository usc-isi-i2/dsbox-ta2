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
from dsbox.schema import TaskType, TaskSubType, Metric
from dsbox.executer.helper import ExecutionHelper

import sklearn.metrics
from sklearn.externals import joblib

class Controller(object):
    """
    This is the overall "planning" coordinator. It is passed in the problem directory
    and the primitives library directory, and it generates plans by calling out to L1, L2
    and L3 planners.
    """
    def __init__(self, directory, libdir, outputdir):
        self.directory = os.path.abspath(directory)
        self.helper = ExecutionHelper(directory, outputdir)

        self.problem = self.helper.load_json(directory + "/problemSchema.json")
        self.task_type = TaskType(self.problem['taskType'])
        self.task_subtype = None
        subtype = self.problem.get('taskSubType', None)
        if subtype:
            subtype = subtype.replace(self.task_type.value.title(), "")
            self.task_subtype = TaskSubType(subtype)

        metricstr = self.problem.get('metric', 'accuracy')
        metricstr = metricstr[0].lower() + metricstr[1:]
        self.metric = Metric(metricstr)
        self.metric_function = self._get_metric_function(self.metric)

        self.schema = self.helper.schema
        self.columns = self.helper.columns
        self.targets = self.helper.targets
        self.indexcol = self.helper.indexcol
        self.media_type = self.helper.media_type

        self.train_data = self.helper.read_data(directory +'/data/trainData.csv.gz',
                                         self.columns, self.indexcol)
        self.train_labels = self.helper.read_data(directory +'/data/trainTargets.csv.gz',
                                           self.targets, self.indexcol)

        self.libdir = os.path.abspath(libdir)
        self.outputdir = os.path.abspath(outputdir)
        if os.path.exists(self.outputdir):
            shutil.rmtree(self.outputdir)
        os.makedirs(self.outputdir+"/models")
        os.makedirs(self.outputdir+"/executables")

        self.logfile = open("%s/log.txt" % self.outputdir, 'w')
        self.errorfile = open("%s/stderr.txt" % self.outputdir, 'w')
        self.pipelinesfile = open("%s/pipelines.txt" % self.outputdir, 'w')

        # Redirect stderr to error file
        sys.stderr = self.errorfile

        self.l1_planner = LevelOnePlannerProxy(self.libdir, self.task_type, self.task_subtype, self.media_type)
        self.l2_planner = LevelTwoPlanner(self.libdir, self.helper)

    def convert_l1_to_l2(self, pipeline):
        pipeline.get_primitives()

    def show_status(self, status):
        sys.stdout.write("%s\n" % status)
        sys.stdout.flush()

    def start(self, cutoff=10):
        self.logfile.write("Task type: %s\n" % self.task_type)
        self.logfile.write("Metric: %s\n" % self.metric)

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

            self.logfile.write("\nL2 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_pipelines))

            self.show_status("Found %d executable pipeline(s). Testing them..." % len(l2_pipelines))

            for l2_pipeline in l2_pipelines:
                expipe = self.l2_planner.patch_and_execute_pipeline(
                        l2_pipeline, df, df_lbl, self.columns, self.task_type, self.metric, self.metric_function)
                l2_pipelines_handled[str(l2_pipeline)] = True
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

