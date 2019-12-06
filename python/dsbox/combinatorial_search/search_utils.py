import bisect
import enum
import pickle
import operator
import os
import random
from d3m.metadata.base import ALL_ELEMENTS
from d3m.base import utils
from d3m.metadata.base import DataMetadata

comparison_metrics = ['training_metrics', 'cross_validation_metrics', 'test_metrics']

def random_choices_without_replacement(population, weights, k=1):
    """
    Randomly sample multiple element based on weights witout replacement.
    """
    assert len(weights) == len(population)
    if k > len(population):
        k = len(population)
    weights = list(weights)
    result = []
    for index in range(k):
        cum_weights = list(accumulate(weights))
        total = cum_weights[-1]
        i = bisect.bisect(cum_weights, random.random() * total)
        result.append(population[i])
        weights[i] = 0
    return result


def accumulate(iterable, func=operator.add):
    """
    Sum all the elments
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total


def save_pickled_dataset(dataset, dataset_name):
    base_dir = os.environ.get("D3MLOCALDIR", "/tmp")
    dataset_path = os.path.join(base_dir, dataset_name + ".pkl")
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)


def load_pickled_dataset(dataset_name):
    base_dir = os.environ.get("D3MLOCALDIR", "/tmp")
    dataset_path = os.path.join(base_dir, dataset_name + ".pkl")
    if not os.path.exists(dataset_path):
        return None
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)


class Status(enum.Enum):
    OK = 0
    PROBLEM_NOT_IMPLEMENT = 148
