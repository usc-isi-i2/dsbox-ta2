"""D3M Dataset Schema Version 2.07
"""

from enum import Enum

# D3M Dataset Annotation Schema 2.07

class VariableFileType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    GRAPH = "graph"
    TABULAR = "table"
    TIMESERIES = "timeseries"
    NONE = "none"
