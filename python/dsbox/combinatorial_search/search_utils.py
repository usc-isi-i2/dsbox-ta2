import bisect
import enum
import operator
import random
from d3m.metadata.base import ALL_ELEMENTS

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


def get_target_columns(dataset: 'Dataset', problem_doc_metadata: 'Metadata'):
    problem = problem_doc_metadata.query(())["inputs"]["data"]
    datameta = dataset.metadata
    target = problem[0]["targets"]
    resID = target[0]["resID"]
    colIndex = target[0]["colIndex"]
    datalength = datameta.query((resID, ALL_ELEMENTS,))["dimension"]['length']
    targetlist = []
    for v in range(datalength):
        types = datameta.query((resID, ALL_ELEMENTS, v))["semantic_types"]
        for t in types:
            if t == 'https://metadata.datadrivendiscovery.org/types/PrimaryKey':
                targetlist.append(v)
    targetlist.append(colIndex)
    targetcol = dataset[resID].iloc[:, targetlist]
    return targetcol


class Status(enum.Enum):
    OK = 0
    PROBLEM_NOT_IMPLEMENT = 148
