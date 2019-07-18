import logging
import traceback

import pandas

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base
from d3m.metadata.problem import PerformanceMetric

__all__ = ['ColumnRole', 'ResourceType', 'ColumnType', 'SpecializedProblem', 'get_target_columns', 'larger_is_better']

_logger = logging.getLogger(__name__)

def larger_is_better(metric_spec) -> bool:
    '''
    Retruns true, if larger metric value is better, such as for ACCURACY and F1_MICRO.
    Moved from dsbox/pipeline/utils.py
    '''
    if isinstance(metric_spec, str):
        metric_name = metric_spec
        metric = PerformanceMetric.get_map()[metric_name]
        print('Should not use "str" for metric')
        traceback.print_stack()
    elif isinstance(metric_spec, dict):
        metric_name = metric_spec['metric']
        metric = PerformanceMetric.get_map()[metric_name]
        print('Should not use "str" for metric (2)')
        traceback.print_stack()
    elif isinstance(metric_spec, PerformanceMetric):
        metric = metric_spec
    else:
        raise ValueError(f'metric spec not recognized: {metric_sepc}')

    return metric.best_value() > metric.worst_value()

def get_target_columns(dataset: container.Dataset):
    """
    Extracts true targets from the Dataset's entry point, or the only tabular resource.
    It requires that there is only one primary index column, which it makes the first
     column, named ``d3mIndex``. Then true target columns follow.

    We return a regular Pandas DataFrame with column names matching those in the metadata.
    We convert all columns to strings to match what would be loaded from ``predictions.csv`` file.
    It encodes any float vectors as strings.

    From: d3m/contrib/primitives/compute_scores.py:ComputeScoresPrimitive._get_truth
    """

    main_resource_id, main_resource = base_utils.get_tabular_resource(dataset, None, has_hyperparameter=False)

    # We first copy before modifying in-place.
    main_resource = container.DataFrame(main_resource, copy=True)
    main_resource = _encode_columns(main_resource)

    dataframe = _to_dataframe(main_resource)

    indices = list(dataset.metadata.get_index_columns(at=(main_resource_id,)))
    targets = list(dataset.metadata.list_columns_with_semantic_types(
        ['https://metadata.datadrivendiscovery.org/types/TrueTarget'],
        at=(main_resource_id,),
    ))

    if not indices:
        raise exceptions.InvalidArgumentValueError("No primary index column.")
    elif len(indices) > 1:
        raise exceptions.InvalidArgumentValueError("More than one primary index column.")
    if not targets:
        raise ValueError("No true target columns.")

    dataframe = dataframe.iloc[:, indices + targets]

    dataframe = dataframe.rename({dataframe.columns[0]: 'd3mIndex'})

    if 'confidence' in dataframe.columns[1:]:
        raise ValueError("True target column cannot be named \"confidence\". It is a reserved name.")
    if 'd3mIndex' in dataframe.columns[1:]:
        raise ValueError("True target column cannot be named \"d3mIndex\". It is a reserved name.")

    if d3m_utils.has_duplicates(dataframe.columns):
        duplicate_names = list(dataframe.columns)
        for name in set(dataframe.columns):
            duplicate_names.remove(name)
        raise exceptions.InvalidArgumentValueError(
            "True target columns have duplicate names: {duplicate_names}".format(
                duplicate_names=sorted(set(duplicate_names)),
            ),
        )

    dataframe = container.DataFrame(dataframe)
    dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
    dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'http://schema.org/Integer')
    dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
    return dataframe


def _to_dataframe(inputs: container.DataFrame) -> pandas.DataFrame:
    # We have to copy, otherwise setting "columns" modifies original DataFrame as well.
    dataframe = pandas.DataFrame(inputs, copy=True)

    column_names = []
    for column_index in range(len(inputs.columns)):
        column_names.append(inputs.metadata.query_column(column_index).get('name', inputs.columns[column_index]))

    # Make sure column names are correct.
    dataframe.columns = column_names

    # Convert all columns to string.
    return dataframe.astype(str)

def _encode_columns(inputs: container.DataFrame) -> container.DataFrame:
    """
    Encode numpy arrays of numbers into float vectors.
    """

    outputs = inputs
    target_columns = outputs.metadata.list_columns_with_semantic_types(
        ('https://metadata.datadrivendiscovery.org/types/PredictedTarget',),
    )

    for column_index in target_columns:
        structural_type = outputs.metadata.query_column(column_index).get('structural_type', None)

        if structural_type is None:
            continue

        if not issubclass(structural_type, container.ndarray):
            continue

        new_column = []
        all_strings = True
        for value in outputs.iloc[:, column_index]:
            assert isinstance(value, container.ndarray)

            if value.ndim == 1:
                new_column.append(','.join(str(v) for v in value))
            else:
                all_strings = False
                break

        if not all_strings:
            continue

        outputs_metadata = outputs.metadata
        outputs.iloc[:, column_index] = new_column
        outputs.metadata = outputs_metadata.update_column(column_index, {
            'structural_type': str,
            'dimension': metadata_base.NO_VALUE,
        })
        outputs.metadata = outputs.metadata.remove(
            (metadata_base.ALL_ELEMENTS, column_index, metadata_base.ALL_ELEMENTS),
            recursive=True,
        )

    return outputs

class ColumnRole:
    INDEX = 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
    MULTIINDEX = 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey'
    KEY = 'https://metadata.datadrivendiscovery.org/types/UniqueKey'
    ATTRIBUTE = 'https://metadata.datadrivendiscovery.org/types/Attribute'
    SUGGESTED_TARGET = 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'
    TIME_INDICATOR = 'https://metadata.datadrivendiscovery.org/types/Time'
    LOCATION_INDICATOR = 'https://metadata.datadrivendiscovery.org/types/Location'
    BOUNDARY_INDICATOR = 'https://metadata.datadrivendiscovery.org/types/Boundary'
    INTERVAL = 'https://metadata.datadrivendiscovery.org/types/Interval'
    INSTANCE_WEIGHT = 'https://metadata.datadrivendiscovery.org/types/InstanceWeight'
    BOUNDING_POLYGON = 'https://metadata.datadrivendiscovery.org/types/BoundingPolygon'
    SUGGESTED_PRIVILEGED_DATA = 'https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData'
    SUGGESTED_GROUPING_KEY = 'https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey'
    EDGE_SOURCE = 'https://metadata.datadrivendiscovery.org/types/EdgeSource'
    DIRECTED_EDGE_SOURCE = 'https://metadata.datadrivendiscovery.org/types/DirectedEdgeSource'
    UNDIRECTE_DEDGE_SOURCE = 'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeSource'
    SIMPLE_EDGE_SOURCE = 'https://metadata.datadrivendiscovery.org/types/SimpleEdgeSource'
    MULTI_EDGE_SOURCE = 'https://metadata.datadrivendiscovery.org/types/MultiEdgeSource'
    EDGE_TARGET = 'https://metadata.datadrivendiscovery.org/types/EdgeTarget'
    DIRECTED_EDGE_TARGET = 'https://metadata.datadrivendiscovery.org/types/DirectedEdgeTarget'
    UNDIRECTED_EDGE_TARGET = 'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeTarget'
    SIMPLE_EDGE_TARGET = 'https://metadata.datadrivendiscovery.org/types/SimpleEdgeTarget'
    MULTI_EDGE_TARGET = 'https://metadata.datadrivendiscovery.org/types/MultiEdgeTarget'
    #Others
    TARGET = 'https://metadata.datadrivendiscovery.org/types/Target'
    TRUE_TARGET = 'https://metadata.datadrivendiscovery.org/types/TrueTarget'
    PRIVILEGED_DATA = 'https://metadata.datadrivendiscovery.org/types/PrivilegedData'
    REDACTED_PRIVILEGED_DATA = 'https://metadata.datadrivendiscovery.org/types/RedactedPrivilegedData'


class ResourceType:
    # File collections.
    IMAGE = 'http://schema.org/ImageObject'
    VIDEO = 'http://schema.org/VideoObject'
    AUDIO = 'http://schema.org/AudioObject'
    TEXT = 'http://schema.org/Text'
    SPEECH = 'https://metadata.datadrivendiscovery.org/types/Speech'
    TIMESERIES = 'https://metadata.datadrivendiscovery.org/types/Timeseries'
    RAW = 'https://metadata.datadrivendiscovery.org/types/UnspecifiedStructure'
    # Other
    GRAPH = 'https://metadata.datadrivendiscovery.org/types/Graph'
    EDGE_LIST = 'https://metadata.datadrivendiscovery.org/types/EdgeList'
    TABLE = 'https://metadata.datadrivendiscovery.org/types/Table'


class ColumnType:
   BOOLEAN = 'http://schema.org/Boolean'
   INTEGER = 'http://schema.org/Integer'
   REAL = 'http://schema.org/Float'
   STRING = 'http://schema.org/Text'
   CATEGORICAL = 'https://metadata.datadrivendiscovery.org/types/CategoricalData'
   DATE_TIME = 'http://schema.org/DateTime'
   REAL_VECTOR = 'https://metadata.datadrivendiscovery.org/types/FloatVector'
   JSON = 'https://metadata.datadrivendiscovery.org/types/JSON'
   GEOJSON = 'https://metadata.datadrivendiscovery.org/types/GeoJSON'
   UNKNOWN = 'https://metadata.datadrivendiscovery.org/types/UnknownType'


class SpecializedProblem:
    PRIVILEGED_INFORMATION = "PrivilegedInformation"
    SEMI_SUPERVISED = "SemiSupervised"
    NONE = "None"
