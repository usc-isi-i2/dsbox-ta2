from d3m.metadata.problem import PerformanceMetric

__all__ = ['ColumnRole', 'ResourceType', 'ColumnType', 'SpecializedProblem', 'larger_is_better']


def larger_is_better(metric_spec) -> bool:
    '''
    Retruns true, if larger metric value is better, such as for ACCURACY and F1_MICRO.
    Moved from dsbox/pipeline/utils.py
    '''
    if isinstance(metric_spec, str):
        metric_name = metric_spec
    else:
        metric_name = metric_spec['metric']
    metric = PerformanceMetric.get_map()[metric_name]
    return metric.best_value() > metric.worst_value()


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
