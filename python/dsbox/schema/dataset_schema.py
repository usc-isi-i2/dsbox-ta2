"""D3M Dataset Schema Version 2.07
"""

from enum import Enum

# D3M Dataset Annotation Schema 
# https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/tree/shared/schemas
# Updated 1/27/2018

class VariableFileType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SPEECH = "speech"
    TEXT = "text"
    GRAPH = "graph"
    TABULAR = "table"
    TIMESERIES = "timeseries"
    NONE = "none"

class ColumnRole(Enum):
    INDEX = "index"
    KEY = "key"
    ATTRIBUTE = "attribute"
    SUGGESTED_TARGET = "suggestedTarget"
    TIME_INDICATOR = "timeIndicator"
    LOCATION_INDICATOR = "locationIndicator"
    BOUNDARY_INDICATOR = "boundaryIndicator"
    INSTANCE_WEIGHT = "instanceWeight"

class ColumnType(Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    REAL = "real"
    STRING = "string"
    CATEGORICAL = "categorical"
    DATETIME = "dateTime"
