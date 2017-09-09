import os
import sys
import os.path
import uuid
import math
import numpy
import shutil
import pandas as pd

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.schema.data_profile import DataProfile
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.executer.executionhelper import ExecutionHelper

import sklearn.metrics
from sklearn.metrics import make_scorer
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
    def __init__(self, libdir):
        self.libdir = os.path.abspath(libdir)
        self.columns = []
        self.targets = []
        self.train_data = None
        self.train_labels = None
        self.indexcol = None
        self.media_type = None
        self.exec_pipelines = []

<<<<<<< Updated upstream
        self.l1_planner = None
        self.l2_planner = None

        self.libdir = os.path.abspath(libdir)
        self.outputdir = os.path.abspath(outputdir)
        if os.path.exists(self.outputdir):
            shutil.rmtree(self.outputdir)
        os.makedirs(self.outputdir + os.sep + "models")
        os.makedirs(self.outputdir + os.sep + "executables")

        self.logfile = open("%s%slog.txt" % (self.outputdir, os.sep), 'w')
        self.errorfile = open("%s%sstderr.txt" % (self.outputdir, os.sep), 'w')
        self.pipelinesfile = open("%s%spipelines.txt" % (self.outputdir, os.sep), 'w')
=======
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
        self.helper = ExecutionHelper(self.train_dir, self.exec_dir, None, self.data_schema)
        self.columns = self.helper.columns
        self.targets = self.helper.targets
        self.indexcol = self.helper.indexcol
        self.media_type = self.helper.media_type
        self.train_data = self.helper.read_data(self.train_dir + os.sep + 'trainData.csv.gz',
                                                 self.columns, self.indexcol)
        self.train_labels = self.helper.read_data(self.train_dir + os.sep + 'trainTargets.csv.gz',
                                                 self.targets, self.indexcol)

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
            helper = ExecutionHelper(self, data_directory, self.exec_dir)
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
            helper = ExecutionHelper(self, data_directory, self.exec_dir)
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
    Set the task type, metric and output type via the schema
    """
    def load_problem_schema(self):
        self.problem = self.helper.load_json(self.problem_schema)
        self.set_task_type(
            self.problem.get('taskType', None),
            self.problem.get('taskSubType', None)
        )
        self.set_metric(self.problem.get('metric', 'accuracy'))
        self.set_output_type(None)

    """
    Set the task type and task subtype
    """
    def set_task_type(self, task_type, task_subtype=None):
        self.task_type = TaskType(task_type)
        self.task_subtype = None
        if task_subtype is not None:
            task_subtype = task_subtype.replace(self.task_type.value.title(), "")
            self.task_subtype = TaskSubType(task_subtype)

    """
    Set the metric
    """
    def set_metric(self, metric):
        metric = metric[0].lower() + metric[1:]
        self.metric = Metric(metric)
        self.metric_function = self._get_metric_function(self.metric)

    """
    Set the output type
    """
    def set_output_type(self, output_type):
        self.output_type = output_type

    """
    Initialize the L1 and L2 planners
    """
    def initialize_planners(self):
        self.l1_planner = LevelOnePlannerProxy(self.libdir, self.task_type, self.task_subtype, self.media_type)
        self.l2_planner = LevelTwoPlanner(self.libdir, self.helper)

    """
    Train and select pipelines
    """
    def train(self, planner_event_handler, cutoff=10):
        self.logfile.write("Task type: %s\n" % self.task_type)
        self.logfile.write("Metric: %s\n" % self.metric)

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
                            l2_l1_map[str(l2_pipeline)] = l1_pipeline
                            l2_pipelines.append(l2_pipeline)
                            yield pe.SubmittedPipeline(l2_pipeline)

            self.logfile.write("\nL2 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_pipelines))

            self._show_status("Found %d executable pipeline(s). Testing them..." % len(l2_pipelines))

            for l2_pipeline in l2_pipelines:
                yield pe.RunningPipeline(l2_pipeline)

                # TODO: Execute parallelly (fork, or separate thread)
                expipe = self.l2_planner.patch_and_execute_pipeline(
                        l2_pipeline, df, df_lbl, self.columns, self.task_type, self.metric, self.metric_function)
                l2_pipelines_handled[str(l2_pipeline)] = True

                yield pe.CompletedPipeline(l2_pipeline, expipe)

                if expipe:
                    self.exec_pipelines.append(expipe)

            self.exec_pipelines = sorted(self.exec_pipelines, key=lambda x: self._sort_by_metric(x))

            self.logfile.write("\nL2 Executed Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(self.exec_pipelines))

            # TODO: Do Pipeline Hyperparameter Tuning

            # Pick top N pipelines, and get similar pipelines to it from the L1 planner to further explore
            l1_related_pipelines = []
            for index in range(0, cutoff):
                if index >= len(self.exec_pipelines):
                    break
                pipeline = l2_l1_map.get(str(self.exec_pipelines[index][0]))
                if pipeline:
                    related_pipelines = self.l1_planner.get_related_pipelines(pipeline)
                    for related_pipeline in related_pipelines:
                        if not l1_pipelines_handled.get(str(related_pipeline), False):
                            l1_related_pipelines.append(related_pipeline)


            self.logfile.write("\nRelated L1 Pipelines to top %d L2 Pipelines:\n-------------\n" % cutoff)
            self.logfile.write("%s\n" % str(l1_related_pipelines))
            l1_pipelines = l1_related_pipelines

        # Ended planners
        self._show_status("Found total %d successfully executing pipeline(s)..." % len(self.exec_pipelines))

        # Create executables
        self.pipelinesfile.write("# Pipelines ranked by metric (%s)\n" % self.metric)
        for index in range(0, len(self.exec_pipelines)):
            pipeline = self.exec_pipelines[index][0]
            self.pipelinesfile.write("%s ( %s ) : %2.4f\n" % (pipeline.id, pipeline, self.exec_pipelines[index][1]))
            self.helper.create_pipeline_executable(pipeline, pipeline.id)

    '''
    Predict results on test data given a pipeline
    '''
    def test(self, pipeline, test_directory):
        helper = ExecutionHelper(self, test_directory, self.exec_dir, 'testData.csv.gz')
        testdf = helper.data
        print("** Evaluating pipeline %s" % str(pipeline))
        for primitive in pipeline.primitives:
            # Initialize primitive
            print("Executing %s" % primitive)
            executables = primitive.executables
            if primitive.task == "Modeling":
                result = pd.DataFrame(executables.predict(testdf), columns=["prediction"])
                resultfile = "%s%s%s.csv" % (self.tmp_dir, os.sep, str(uuid.uuid1()))
                result.to_csv(resultfile, index_label="d3mIndex")
                return resultfile
            elif primitive.task == "PreProcessing":
                profile = DataProfile(testdf)
                testdf = helper.execute_primitive(primitive, testdf, None, profile)
            elif primitive.task == "FeatureExtraction":
                testdf = helper.featurise(testdf, primitive, True, primitive.is_persistent)
        return None

    def stop(self):
        '''
        Stop planning, and write out the current list (sorted by metric)
        '''


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

    def _get_arg_value(self, arg_specification, task_type, metric):
        if not arg_specification.startswith('*'):
            return arg_specification
        arg_specification = arg_specification[1:]
        metric_func = self._get_metric_function(metric)
        if arg_specification == "SCORER":
            return make_scorer(metric_func, greater_is_better=True)
        elif arg_specification == "LOSS":
            return make_scorer(metric_func, greater_is_better=False)
        elif arg_specification == "ESTIMATOR":
            if task_type == TaskType.CLASSIFICATION:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression()
            elif task_type == TaskType.REGRESSION:
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            else:
                raise Exception("Not yet implemented: Arg specification ESTIMATOR task type: {}"
                                .format(task_type))
        else:
            raise Exception(
                "Unkown Arg specification: {}".format(arg_specification))
