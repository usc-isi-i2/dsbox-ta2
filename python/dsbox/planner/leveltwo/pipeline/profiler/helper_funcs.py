import re
from dateutil.parser import parse

"""
this script contains all the helper functions that apply to a input string
most refer from: https://github.com/usc-isi-i2/dptk

"""

def convertAlphatoNum(input):

    non_decimal = re.compile(r'[^\d\.]+')
    return non_decimal.sub(' ', input)

def is_Integer_Number_Ext(s):
    """
        return any(char.isdigit() for char in inputString)
    """ 
    try:
        int(s)
        return True
    except:
        try:
            int(convertAlphatoNum(s))
            return True
        except:
            return False


def is_Decimal_Number_Ext(s):
    try:
        float(s)
        return True
    except:
        try:
            float(convertAlphatoNum(s))
            return True
        except:
            return False
        
def is_Integer_Number(s):
    # return any(char.isdigit() for char in inputString)
    try:
        int(s)
        return True
    except:
        return False


def is_Decimal_Number(s):
    try:
        float(s)
        return True
    except:
        return False

def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False

def getDecimal(s):
    try:
        return float(s)
    except:
        try:
            return float(convertAlphatoNum(s))
        except:
            return 0.0
