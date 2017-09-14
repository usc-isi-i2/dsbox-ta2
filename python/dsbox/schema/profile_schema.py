from enum import Enum

class DataProfileType(Enum):
    '''
    Data profiles
    '''
    MISSING_VALUES = "Missing Values"
    NUMERICAL = "Numerical"
    UNIQUE = "Unique"
    NEGATIVE = "Negative"
    TEXT = "Text"
