
import os
import sys
sys.path.insert(0, '/nas/home/kyao/kyao-repo/dsbox/dsbox-ta2/python')

from sklearn.externals import joblib


from dsbox.planner.common.pipeline import Pipeline
from dsbox.schema.problem_schema import Metric
from dsbox.planner.common.resource_manager import PipelineExecStat

DISSIMILARITY_METRICS = set([Metric.MEAN_SQUARED_ERROR, Metric.ROOT_MEAN_SQUARED_ERROR, Metric.ROOT_MEAN_SQUARED_ERROR_AVG,
                        Metric.MEAN_ABSOLUTE_ERROR, Metric.EXECUTION_TIME])

def load_run_stats(output_root_dir):
    '''Load all stat files under output_root_dir'''
    stat = dict()
    for output_dir in os.listdir(output_root_dir):
        astat = load_run_stat(os.path.join(output_root_dir, output_dir))
        if astat is not None:
            stat[output_dir] = astat
    return stat

def load_run_stat(output_dir):
    stat = None
    pipeline_files = os.path.join(output_dir, 'temp', 'statistics.pkl')
    if os.path.exists(pipeline_files):
        stat = joblib.load(pipeline_files)
    return stat

def smaller_is_better(metric: Metric):
    return metric in DISSIMILARITY_METRICS

def larger_is_better(metric: Metric):
    return not(metric in DISSIMILARITY_METRICS)

def best_pipeline_info(run_stat, name):
    pipeline_stats: PipelineExecStat = run_stat.pipelines.values()
    pipelines = [s.pipeline for s in pipeline_stats if s.pipeline.planner_result is not None]

    rows = []

    if not pipelines:
        return rows

    metrics = [Metric[name] for name in pipelines[0].planner_result.metric_values.keys()]

    for metric in metrics:
        pipelines_sorted = sorted(pipelines, key=lambda p: p.planner_result.metric_values[metric.name], reverse=larger_is_better(metric))
        best = pipelines_sorted[0]
        is_ensemble = best.ensemble is not None
        if is_ensemble:
            is_default_hyperparams = True
            for pipeline in best.ensemble.pipelines:
                if pipeline.primitives[-1]._hyperparams is not None:
                    is_default_hyperparams = False
                    break
            for pipeline in pipelines_sorted:
                if pipeline.primitives and pipeline.primitives[-1]._hyperparams is None:
                    best_default_hyperparams = pipeline
                    break
        else:
            is_default_hyperparams = best.primitives[-1]._hyperparams is None
            if is_default_hyperparams:
                best_default_hyperparams = best
            else:
                for pipeline in pipelines_sorted:
                    if pipeline.primitives and pipeline.primitives[-1]._hyperparams is None:
                        best_default_hyperparams = pipeline
                        break
        row = [metric.name, best.planner_result.metric_values[metric.name], best_default_hyperparams.planner_result.metric_values[metric.name],
               not is_default_hyperparams, is_ensemble, name]
        rows.append(row)
    return rows


stats = load_run_stats('/nas/home/kyao/dsbox/runs/output-event-loop-3')
all_rows = []
for problem_name, stat in stats.items():
    all_rows = all_rows + best_pipeline_info(stat, problem_name)

print()
print('metric, best, best_no_hyperparams, best_uses_hyperparam, best_uses_ensemble, problem')
for row in all_rows:
    print('{}, {:8.4f}, {:8.4f}, {}, {}, {}'.format(*row))
