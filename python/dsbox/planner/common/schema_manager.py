import os
import json

from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric

# sklearn metric functions
import sklearn.metrics

DEFAULT_PROBLEM_SCHEMA = "problemSchema.json"

class SchemaManager(object):

    def __init__(self):
        self.problem_id = None
        self.task_type = None
        self.task_subtype = None
        self.metrics = []
        self.metric_functions = []
        self.output_type = None

    """
    Set the task type, metric and output type via the schema
    """
    def load_problem_schema(self, problem_schema):
        problem = self.load_json(problem_schema)
        self.problem_id = problem['problemId']
        taskType = problem.get('taskType', None)
        # FIXME: treating collaborativeFiltering as regression for now
        if taskType == "collaborativeFiltering":
            taskType = "regression"
        self.set_task_type(
            taskType,
            problem.get('taskSubType', None)
        )
        self.set_metric(problem.get('metric', 'accuracy'))
        self.set_output_type(None)

    def load_json(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d

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
    Set the output type
    """
    def set_output_type(self, output_type):
        self.output_type = output_type

    """
    Set the metric
    """
    def set_metric(self, metric):
        metric = metric[0].lower() + metric[1:]
        self.metrics = [Metric(metric)]
        self.set_metric_functions()

    def set_metrics(self, metrics):
        self.metrics = metrics
        self.set_metric_functions()

    def set_metric_functions(self):
        self.metric_functions = []
        for metric in self.metrics:
            self.metric_functions.append(self._get_metric_function(metric))

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
        elif metric==Metric.MEAN_SQUARED_ERROR:
            return sklearn.metrics.mean_squared_error
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
