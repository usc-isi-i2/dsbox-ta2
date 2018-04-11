
import json
import os
import re
import sys
sys.path.insert(0, '/nas/home/kyao/kyao-repo/dsbox/dsbox-ta2/python')

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List

import numpy as np
import pandas as pd

from dsbox.schema.problem_schema import Metric

def filter_key(dict_list: List[dict], key, value=None, substring=True):
    '''Return list of dict that satisfies the key value criteria'''
    if value is None:
        return [d for d in dict_list if key in d]
    elif substring:
        return [d for d in dict_list if key in d and value in d[key]]
    return [d for d in dict_list if key in d and value == d[key]]

def split_key(dict_list: List[dict], key):
    '''Return list of lists of dict, split by value of key'''
    result = defaultdict(list)
    for d in dict_list:
        if key in d:
            result[d[key]].append(d)
    return result

def grep(fin, substring, invert=False):
    '''Return lines containing substring'''
    for line in fin:
        if substring in line:
            if not invert:
                yield line
        else:
            if invert:
                yield line

def egrep(fin, pattern):
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    for line in fin:
        if re.search(pattern, line):
            yield line

def get_learner_info(info):
    '''Returns dict describing the classifier/regresser primitive'''
    if info['primitives'] is None:
        return None
    if len(info['primitives']) > 1:
        return info['primitives'][-2]
    return None

def get_learner_class(info):
    '''Returns class name of the classifier/regresser primitive'''
    if info['primitives'] is None:
        return None
    if len(info['primitives']) > 1:
        return info['primitives'][-2]['class']
    return None

class BestPipeline():
    def __init__(self, metric_type='training_metric'):
        self.metric_type = metric_type
        self.best_by_problem = defaultdict(dict)

    def process(self, line):
        if isinstance(line,str):
            info = json.loads(line)
        else:
            info = line
        if not 'pipe_info' in info.keys():
            return
        if info[self.metric_type] is None:
            return

        problem = info['problem_id']
        for metric_name in info[self.metric_type].keys():
            metric = Metric[metric_name]
            if metric in self.best_by_problem[problem]:
                if metric.larger_is_better():
                    if info[self.metric_type][metric_name] > self.best_by_problem[problem][metric][self.metric_type][metric_name]:
                        self.best_by_problem[problem][metric] = info
                else:
                    if info[self.metric_type][metric_name] < self.best_by_problem[problem][metric][self.metric_type][metric_name]:
                        self.best_by_problem[problem][metric] = info
            else:
                self.best_by_problem[problem][metric] = info

    def get_result(self):
        rows = []
        for problem, metrics in self.best_by_problem.items():
            for metric, info in metrics.items():
                if isinstance(info, list):
                    print(info)
                is_ensemble = len(info['ensemble']) > 1
                if is_ensemble:
                    learner_name = 'ensemble'
                    is_default_hyperparams = False
                else:
                    learner = get_learner_info(info)
                    learner_name = learner['class'].split('.')[-1]
                    is_default_hyperparams = learner['hyperparams'] is None
                row = [metric.name, info[self.metric_type][metric.name], not is_default_hyperparams, is_ensemble, learner_name, problem]
                rows.append(row)
        return rows

def print_best_pipeline(fin):
    '''For each dataset print best pipeline result and attributes'''
    bp = BestPipeline()
    for line in fin:
        bp.process(line)
    all_rows = bp.get_result()
    print()
    print('{:>25}, {:>8}, {:>5}, {:>5}, {}, {}'.format(
        'metric', 'best', 'best_uses_hyperparam', 'best_uses_ensemble', 'learner', 'problem'))
    for row in all_rows:
        print('{:>25}, {:8.4f}, {:5}, {:5}, {}, {}'.format(*row))

def gather_metric_values(fin, metric_type='training_metric'):
    learner_values = []
    for line in fin:
        if isinstance(line, dict):
            info = line
            if not 'pipe_info' in info.keys():
                continue
        else:
            if not 'pipe_info' in line:
                continue
            info = json.loads(line)
        if len(info['ensemble']) > 1:
            continue
        if info[metric_type] is not None:
            if len(info[metric_type]) > 1:
                raise ValueError('Multiple metrics found')
            for metric, value in info[metric_type].items():
                learner_values.append(value)
    return learner_values, metric

def gather_metric_values_by_learner(fin, metric_type='training_metric'):
    failed = set()
    metric = None
    learner_values = defaultdict(list)
    for line in fin:
        if isinstance(line, dict):
            info = line
            if not 'pipe_info' in info.keys():
                continue
        else:
            if not 'pipe_info' in line:
                continue
            info = json.loads(line)
        if len(info['ensemble']) > 1:
            continue
        learner = get_learner_class(info)
        if info[metric_type] is None:
            failed.add(learner)
        else:
            if len(info[metric_type]) > 1:
                raise ValueError('Multiple metrics found')
            for metric, value in info[metric_type].items():
                learner_values[learner].append(value)

    # Since some piplines sucedded, probably failed because of timeout
    # print('Failed: {}'.format(failed))
    failed = failed - set(learner_values.keys())

    return learner_values, metric, failed

def gen_plot_files(fin, plot_file_pattern='output-{}.pdf'):
    all_stats = [json.loads(line) for line in fin if 'pipe_info' in line]
    by_runs = split_key(all_stats, 'run_id')
    pd.set_option('display.width', 160)
    for run_id, stats in by_runs.items():
        print_best_pipeline(stats)

        plot_file = plot_file_pattern.format(run_id)
        pp = PdfPages(plot_file)

        for problem_name, stat in split_key(stats, 'problem_id').items():
            print('** ', problem_name)
            learner_values, metric, failed = gather_metric_values_by_learner(stat)

            d = learner_describe(learner_values)
            print(d.transpose())

            order_by_max = metric and not 'ERROR' in metric

            plt.figure()
            title = '{} (run {})'.format(problem_name, run_id)
            learner_boxplot(learner_values, title=title, order_by_max=order_by_max, xlabel=metric, savepdf=pp)
            print()
        pp.close()


def learner_describe(learner_values):
    '''Returns DataFrame with stats (count, mean, std, min, max, and quartiles)'''
    df = pd.DataFrame()
    for learner, values in learner_values.items():
        df[learner] = pd.Series(values).describe()

    return df

def learner_boxplot(learner_values, order_by_max=True, title=None, xlabel=None,
                    savepdf: PdfPages = None):
    """Give a dataset run, generate box-wisker plot. y-axis correspond to
    algorithms (one box-wisker), and x-axis is the metric

    """
    values = list(learner_values.values())
    labels = list(learner_values.keys())
    labels = ['{} ({})'.format(label, len(values)) for label, values in zip(labels, values)]
    if order_by_max:
        extreme = [max(values) for values in values]
        other_extreme = [min(values) for values in values]
        index = np.argsort(extreme)
    else:
        extreme = [min(values) for values in values]
        index = np.argsort(extreme)[::-1]
        other_extreme = sorted([max(values) for values in values])[::-1]
        while len(other_extreme) > 2 and other_extreme[0] > 10 * other_extreme[1]:
            other_extreme = other_extreme[1:]
            plt.xlim(xmax=1.1*other_extreme[0])


    print('other', other_extreme)
    values_sorted = [values[i] for i in index]
    labels_sorted = [labels[i] for i in index]

    plt.boxplot(values_sorted, vert=False)
    plt.yticks(range(1,len(labels_sorted)+1), labels_sorted)
    plt.subplots_adjust(left=0.35)
    if title:
        plt.title(title)
    if xlabel:
        print(xlabel, type(xlabel))
        plt.xlabel(xlabel)
    if savepdf:
        savepdf.savefig()
    else:
        plt.show()

def learner_hist(learner_values, order_by_max=True, title=None, xlabel=None,
                    savepdf: PdfPages = None):
    """Generate one histogram plot"""
    values = list(learner_values.values())
    if len(values)==0:
        print('Not data. Skipping "{}"'.format(title))
        return

    labels = list(learner_values.keys())
    labels = ['{} ({})'.format(label, len(values)) for label, values in zip(labels, values)]
    if order_by_max:
        extreme = [max(values) for values in values]
        other_extreme = [min(values) for values in values]
        index = np.argsort(extreme)

        max_value = max(extreme)
        min_value = min(other_extreme)
    else:
        extreme = [min(values) for values in values]
        index = np.argsort(extreme)[::-1]
        other_extreme = sorted([max(values) for values in values])[::-1]
        while len(other_extreme) > 2 and other_extreme[0] > 10 * other_extreme[1]:
            print('other', other_extreme)
            other_extreme = other_extreme[1:]
            max_value = 1.1*other_extreme[0]
        min_value = min(extreme)

    # Note: This does not seem to work for subplots with histogram
    plt.xlim(xmin=min_value, xmax=max_value)

    values_sorted = [values[i] for i in index]
    labels_sorted = [labels[i] for i in index]

    select = [len(x) >= 5 for x in values_sorted]
    fig, axis = plt.subplots(sum(select), 1, sharex=True)

    i = 0
    for values, label, use in zip(values_sorted, labels_sorted, select):
        if not use:
            continue
        axis[i].hist(values)
        axis[i].set_title(label)

        # Setting the xlim here works when showing to screen. But, pdf is broken.
        if savepdf is None:
            axis[i].set_xlim([min_value, max_value])

        i += 1

    print(title, plt.xlim())

    if title:
        fig.suptitle(title)
    if xlabel:
        print(xlabel, type(xlabel))
        fig.xlabel(xlabel)
    if savepdf:
        savepdf.savefig()
    else:
        fig.show()

def gen_hist_files(fin, plot_file_pattern='output-hist-{}.pdf'):
    """Generate histogram plots of run metrics."""
    all_stats = [json.loads(line) for line in fin if 'pipe_info' in line]
    by_runs = split_key(all_stats, 'run_id')
    pd.set_option('display.width', 160)
    pp = None
    for run_id, stats in by_runs.items():

        if plot_file_pattern:
            plot_file = plot_file_pattern.format(run_id)
            pp = PdfPages(plot_file)

        for problem_name, stat in split_key(stats, 'problem_id').items():
            print('** ', problem_name)
            learner_values, metric, failed = gather_metric_values_by_learner(stat)

            order_by_max = metric and Metric[metric].larger_is_better()  # not 'ERROR' in metric
            print(order_by_max)

            plt.figure()
            title = '{} (run {})'.format(problem_name, run_id)
            learner_hist(learner_values, title=title, order_by_max=order_by_max, savepdf=pp)
            # learner_hist(learner_values, title=title, order_by_max=order_by_max)
            print()
        if plot_file_pattern:
            pp.close()

def gen_learner_plots(fin, plot_file='output-by-algo.pdf'):
    """Generate one box-wisker metrics plot  for each learner algorithm."""

    all_stats = [json.loads(line) for line in fin if 'pipe_info' in line]

    pp = PdfPages(plot_file)
    learner_problem_values = defaultdict(lambda: defaultdict(list))
    learner_problem_runs = defaultdict(lambda: defaultdict(list))

    for problem_name, stat in split_key(all_stats, 'problem_id').items():
        print(problem_name)
        learner_values, metric, failed = gather_metric_values_by_learner(stat)
        for learner, values in learner_values.items():
            run = '{}'.format(problem_name)
            learner_problem_values[learner][problem_name].append(values)
            learner_problem_runs[learner][problem_name].append(run)

    for learner, problem_values in learner_problem_values.items():
        print(learner)
        values = []
        ylabels = []
        for problem in problem_values.keys():
            print('  ', problem)
            values += problem_values[problem]
            ylabels += learner_problem_runs[learner][problem]

        plt.figure()
        plt.boxplot(values, vert=False)
        plt.subplots_adjust(left=0.35)
        plt.yticks(range(1,len(ylabels)+1), ylabels)
        plt.title(learner)
        pp.savefig()
        # plt.close()
        # plt.show()
    pp.close()





with open('/nas/home/kyao/dsbox/runs/json-output/output-event-loop-3-to-6.json', 'r') as fin:
  print_best_pipeline(fin)

with open('/nas/home/kyao/dsbox/runs/json-output/output-event-loop-3-to-6.json', 'r') as fin:
  gen_plot_files(fin, '/nas/home/kyao/dsbox/runs/plots/output-{}.pdf')

with open('/nas/home/kyao/dsbox/runs/json-output/output-event-loop-3-to-6.json', 'r') as fin:
    gen_hist_files(fin, '/nas/home/kyao/dsbox/runs/plots/output-hist-{}.pdf')

with open('/nas/home/kyao/dsbox/runs/json-output/output-event-loop-3-to-6.json', 'r') as fin:
    gen_learner_plots(fin, '/nas/home/kyao/dsbox/runs/plots/output-by-algo.pdf')

# Display histogram for 'event-loop-3' run with the following datasets '26_|56_|196_|534_|uu3_|299_'
with open('/nas/home/kyao/dsbox/runs/json-output/output-event-loop-3-to-6.json', 'r') as fin:
    fin2 = grep(fin, 'event-loop-3')
    fin3 = egrep(fin2, '26_|56_|196_|534_|uu3_|299_')
    gen_hist_files(fin3, None)
