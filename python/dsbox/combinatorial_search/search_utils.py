import bisect
import enum
import operator
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


def get_target_columns(dataset: 'Dataset', problem_doc_metadata: 'Metadata'):

    main_resource_id, _ = utils.get_tabular_resource(dataset, None, has_hyperparameter=False)
    targetcol_list = DataMetadata.list_columns_with_semantic_types(dataset.metadata, ['https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey', 'https://metadata.datadrivendiscovery.org/types/TrueTarget'], at=(main_resource_id,))
    targetcol = dataset[main_resource_id].iloc[:, targetcol_list]
    # use common primitive's method instead of this old one
    '''
    targetcol = None
    problem = problem_doc_metadata.query(())["inputs"]["data"]
    datameta = dataset.metadata
    target = problem[0]["targets"]
    resID_list = []
    colIndex_list = []
    targetlist = []
    # sometimes we will have multiple targets, so we need to add a for loop here
    for i in range(len(target)):
        resID_list.append(target[i]["resID"])
        colIndex_list.append(target[i]["colIndex"])
    if len(set(resID_list)) > 1:
        print("[ERROR] Multiple targets in different dataset???")

    datalength = datameta.query((resID_list[0], ALL_ELEMENTS,))["dimension"]['length']

    for v in range(datalength):
        types = datameta.query((resID_list[0], ALL_ELEMENTS, v))["semantic_types"]
        for t in types:
            if t == 'https://metadata.datadrivendiscovery.org/types/PrimaryKey':
                targetlist.append(v)
    for each in targetlist:
        colIndex_list.append(each)
    colIndex_list.sort()
    '''
    
    return targetcol
    

class Status(enum.Enum):
    OK = 0
    PROBLEM_NOT_IMPLEMENT = 148
