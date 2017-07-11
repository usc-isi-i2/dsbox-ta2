import pandas as pd
import numpy as np
import helper_funcs as hf
from collections import OrderedDict
from collections import defaultdict
from collections import Counter

def ordered_dict2(column, k):
    unique,counts = np.unique(column, return_counts=True)
    d = dict(zip(unique,counts))
    return OrderedDict(Counter(d).most_common(k))

def ordered_dict(column, k):
    #d = column.value_counts()[:k].to_dict()
    d = column.value_counts().head(k).to_dict()
    return OrderedDict(sorted(d.items(), key=lambda x: x[1], reverse=True))

def tryConvert(cell):
    """
    convert a cell, if possible, to its supposed type(int, float, string)
    note: type of NaN cell is float
    """
    try:
        return int(cell)
    except ValueError, TypeError:
        try:
            return float(cell)
        except ValueError, TypeError:
            return cell

def numerical_stats(column,num_nonblank,sigma=3):
    """
    calculates numerical statistics
    """
    stats = column.describe()
    idict = {}
    idict["mean"] = stats["mean"]
    idict["standard-deviation"] = stats["std"]
    #temporary ugly patch for std
    #when count=1, make std=0
    if stats["count"]==1: idict["standard-deviation"]= 0
    idict["Q1"] = stats["25%"]
    idict["Q2"] = stats["50%"]
    idict["Q3"] = stats["75%"]
    idict["count"] = int(stats["count"])
    idict["ratio"] = stats["count"]/num_nonblank
    outlier = column[(np.abs(column-stats["mean"])>(sigma*stats["std"]))]
    idict["num_outlier"] = outlier.count()
    idict["num_positive"] = column[column>0].count()
    idict["num_negative"] = column[column<0].count()
    idict["num_0"] = column[column==0].count()
    idict["num_1"] = column[column==1].count()
    idict["num_-1"] = column[column==-1].count()
    
    return idict

def compute_numerics(column, feature):
    """
    computes numerical features of the column:
    # of integers/ decimal(float only)/ nonblank values in the column
    statistics of int/decimal/numerics
    """
    feature["missing"]["num_nonblank"] = column.count()

    if column.dtype.kind in np.typecodes['AllInteger']+'u' and column.count() > 0:
        feature["numeric_stats"]["integer"] = numerical_stats(column,feature["missing"]["num_nonblank"])
    elif column.dtype.kind == 'f' and column.count() > 0:
        feature["numeric_stats"]["decimal"] = numerical_stats(column,feature["missing"]["num_nonblank"])
    
    else:
        convert = lambda v: tryConvert(v)
        col = column.apply(convert, convert_dtype=False)
        #col = pd.to_numeric(column,errors='ignore') #doesn't work in messy column?

        col_nonblank = col.dropna()
        col_int = pd.Series([e for e in col_nonblank if type(e) == int or type(e) == np.int64])
        col_float = pd.Series([e for e in col_nonblank if type(e) == float or type(e) == np.float64])

        if col_int.count() > 0:
            feature["numeric_stats"]["integer"] = numerical_stats(col_int,feature["missing"]["num_nonblank"])

        if col_float.count() > 0:
            feature["numeric_stats"]["decimal"] = numerical_stats(col_float,feature["missing"]["num_nonblank"])

        if "integer" in feature["numeric_stats"] or "decimal" in feature["numeric_stats"]:
            col_num = pd.concat([col_float,col_int])
            feature["numeric_stats"]["numeric"] = numerical_stats(col_num,feature["missing"]["num_nonblank"])

def compute_common_numeric_tokens(column, feature, k):
    """
    compute top k frequent numerical tokens and their counts.
    tokens are integer or floats
    e.g. "123", "12.3"
    """
    col = column.str.split(expand=True).unstack().dropna().values
    token = np.array(filter(lambda x: hf.is_Decimal_Number(x), col))
    if token.size:
        feature["frequent-entries"]["most_common_numeric_tokens"] = ordered_dict2(token, k)

def compute_common_alphanumeric_tokens(column, feature, k):
    """
    compute top k frequent alphanumerical tokens and their counts.
    tokens only contain alphabets and/or numbers, decimals with points not included
    """
    col = column.str.split(expand=True).unstack().dropna().values
    token = np.array(filter(lambda x: x.isalnum(), col))
    if token.size:
        feature["frequent-entries"]["most_common_alphanumeric_tokens"] = ordered_dict2(token, k)

def compute_common_values(column, feature, k):
    """
    compute top k frequent cell values and their counts.
    """
    if column.count() > 0:
        feature["frequent-entries"]["most_common_values"] = ordered_dict(column, k)

def compute_common_tokens(column, feature, k):
    """
    compute top k frequent tokens and their counts.
    currently: tokens separated by white space
    at the same time, count on tokens which contain number(s)
    e.g. "$100", "60F", "123-456-7890"
    note: delimiter = " "
    """
    token = column.str.split(expand=True).unstack().dropna().values
    if token.size:
        feature["frequent-entries"]["most_common_tokens"] = ordered_dict2(token, k)
        cnt = sum([any(char.isdigit() for char in c) for c in token])
        if cnt > 0:
            feature["numeric_stats"]["contain_numeric_token"] = {}
            feature["numeric_stats"]["contain_numeric_token"]["count"] = cnt
            feature["numeric_stats"]["contain_numeric_token"]["ratio"] = float(cnt)/token.size

def compute_common_tokens_by_puncs(column, feature, k):
    """
    tokens seperated by all string.punctuation characters:
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    """
    col = column.dropna().values
    token_nested = [("".join((word if word.isalnum() else " ") for word in char).split()) for char in col]
    token = np.array([item for sublist in token_nested for item in sublist])
    if token.size:
        feature["frequent-entries"]["most_common_tokens_puncs"] = ordered_dict2(token, k)
        dist_cnt = np.unique(token).size
        feature["distinct"]["num_distinct_tokens_puncs"] = dist_cnt
        feature["distinct"]["ratio_distinct_tokens_puncs"] = float(dist_cnt)/token.size
        cnt = sum([any(char.isdigit() for char in c) for c in token])
        if cnt > 0:
            feature["numeric_stats"]["contain_numeric_token_puncs"] = {}
            feature["numeric_stats"]["contain_numeric_token_puncs"]["count"] = cnt
            feature["numeric_stats"]["contain_numeric_token_puncs"]["ratio"] = float(cnt)/token.size

def compute_numeric_density(column, feature):
    """
    compute overall density of numeric characters in the column.
    """
    col = column.dropna().values
    if col.size:
        density = np.array([(sum(char.isdigit() for char in c), len(c)) for c in col])
        digit_total = density.sum(axis=0)
        feature["numeric_stats"]["numeric_density"] = float(digit_total[0])/digit_total[1]

def compute_contain_numeric_values(column, feature):
    """
    caculate # and ratio of cells in the column which contains numbers.
    """
    contain_digits = lambda x: any(char.isdigit() for char in x)
    cnt = column.dropna().apply(contain_digits).sum()
    if cnt > 0:
        feature["numeric_stats"]["contain_numeric"] = {}
        feature["numeric_stats"]["contain_numeric"]["count"] = cnt
        feature["numeric_stats"]["contain_numeric"]["ratio"] = float(cnt)/column.count()
