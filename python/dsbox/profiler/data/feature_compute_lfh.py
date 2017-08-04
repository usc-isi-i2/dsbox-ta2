import pandas as pd
from langdetect import detect
import helper_funcs
import string
import numpy as np
import re

# till now, this file totally compute 16 types of features

def compute_missing_space(column, feature):
    """
    NOTE: this function may change the input column. It will trim all the leading and trailing whitespace.
            if a cell is empty after trimming(which means it only contains whitespaces), 
            it will be set as NaN (missing value), and both leading_space and trailing_space will += 1.

    (1). trim and count the leading space and trailing space if applicable. 
        note that more than one leading(trailing) spaces in a cell will still be counted as 1.
    (2). compute the number of missing value for a given series (column); store the result into (feature)
    """
    leading_space = 0
    trailing_space = 0

    for id in xrange(len(column)): 
        cell = column[id]
    #for cell in column:    # 5x faster loop, but cannot modify the column
        if (pd.isnull(cell)):
            continue
           
        change = False
        trim_leading_cell = re.sub(r"^\s+", "", cell)
        if (trim_leading_cell != cell):
            leading_space += 1
            change = True
        trim_trailing_cell = re.sub(r"\s+$", "", trim_leading_cell)
        if ( (trim_trailing_cell != trim_leading_cell) or len(trim_trailing_cell) == 0):
            trailing_space += 1
            change = True

        # change the origin value in data
        if change:
            if (len(trim_trailing_cell) == 0):
                column[id] = np.nan
            else:
                column[id] = trim_trailing_cell

    feature["missing"]["leading_space"] = leading_space
    feature["missing"]["trailing_space"] = trailing_space
            
    feature["missing"]["num_missing"] = pd.isnull(column).sum()
    


def compute_length_distinct(column, feature, delimiter):
    """
    two tasks because of some overlaping computation:

    (1). compute the mean and std of length for each cell, in a given series (column); 
        mean and std precision: 5 after point
        missing value (NaN): treated as does not exist
    (2). also compute the distinct value and token:
        number: number of distinct value or tokens, ignore the NaN
        ratio: number/num_total, ignore all NaN
    """
    # (1)
    column = column.dropna() # get rid of all missing value
    if (column.size == 0):      # if the column is empty, do nothing
        return
    
    feature["length"] = {} # for character and token
    # 1. for character
    feature["length"]["character"] = {}
    lenth_for_all =  column.apply(len)
    feature["length"]["character"]["average"] = lenth_for_all.mean()
    feature["length"]["character"]["standard-deviation"] = lenth_for_all.std()
    
    # 2. for token
    feature["length"]["token"] = {}
    tokenlized = column.str.split(delimiter, expand=True).unstack().dropna()    # tokenlized Series
    lenth_for_token = tokenlized.apply(len)
    feature["length"]["token"]["average"] = lenth_for_token.mean()
    feature["length"]["token"]["standard-deviation"] = lenth_for_token.std()
    
    # (2)
    feature["distinct"]["num_distinct_values"] = column.nunique()
    feature["distinct"]["ratio_distinct_values"] = feature["distinct"]["num_distinct_values"] / float(column.size)
        # using the pre-computed tokenlized in (1), which is series of all tokens
    feature["distinct"]["num_distinct_tokens"] = tokenlized.nunique()
    feature["distinct"]["ratio_distinct_tokens"] = feature["distinct"]["num_distinct_tokens"] / float(tokenlized.size)
    


def compute_lang(column, feature):
    """
    compute which language(s) it use for a given series (column); store the result into (feature).
    not apply for numbers

    PROBLEMS:
    1. not accurate when string contains digits
    2. not accurate when string is too short
    maybe need to consider the special cases for the above conditions
    """
    column = column.dropna() # ignore all missing value
    if (column.size == 0):      # if the column is empty, do nothing
        return
        
    feature["special_type"]["language"] = {}

    for cell in column:
        if cell.isdigit() or helper_funcs.is_Decimal_Number(cell):
            continue
        else:
            #detecting language
            try:
                language = detect(cell)
                if language in feature["language"]:
                    feature["language"][language] += 1
                else:
                    feature["language"][language] = 1
            except Exception as e:
                print "there is something may not be any language nor number: {}".format(cell)
                pass

def compute_filename(column, feature):
    """
    compute number of cell whose content might be a filename
    """
    column = column.dropna() # ignore all missing value

    filename_pattern = r"^\w+\.[a-z]{1,5}"
    column.str.match(filename_pattern)
    num_filename = column.str.match(filename_pattern).sum()
    feature["special_type"]["num_filename"] = num_filename


def compute_punctuation(column, feature, weight_outlier):
    """
    compute the statistical values related to punctuations, for details, see the format section of README.

    not apply for numbers (eg: for number 1.23, "." does not count as a punctuation)

    weight_outlier: = number_of_sigma in function "helper_outlier_calcu"
    """

    column = column.dropna() # get rid of all missing value
    if (column.size == 0):      # if the column is empty, do nothing
        return

    number_of_chars =  sum(column.apply(len))   # number of all chars in column
    num_chars_cell = np.zeros(column.size)   # number of chars for each cell
    puncs_cell = np.zeros([column.size, len(string.punctuation)], dtype=int) # (number_of_cell * number_of_puncs) sized array
    
    # step 1: pre-calculations
    cell_id = -1
    for cell in column:
        cell_id += 1
        num_chars_cell[cell_id] = len(cell)
        # only counts puncs for non-number cell
        if cell.isdigit() or helper_funcs.is_Decimal_Number(cell):  
            continue
        else:
            counts_cell_punc = np.asarray(list(cell.count(c) for c in string.punctuation))
            puncs_cell[cell_id] = counts_cell_punc
            
    counts_column_punc = puncs_cell.sum(axis=0) # number of possible puncs in this column
    cell_density_array = puncs_cell / num_chars_cell.reshape([column.size, 1])
    puncs_density_average = cell_density_array.sum(axis=0) / column.size

    # step 2: extract from pre-calculated data
    # only create this feature when punctuations exist
    if (sum(counts_column_punc) > 0):
        if ("frequent-entries" not in feature.keys()):
            feature["frequent-entries"] = {}
        feature["frequent-entries"]["most_common_punctuations"] = {}

        # extract the counts to feature, for each punctuation
        for i in xrange(len(string.punctuation)):
            if (counts_column_punc[i] == 0):    # if no this punctuation occur in the whole column, ignore
                continue
            else:
                feature["frequent-entries"]["most_common_punctuations"][string.punctuation[i]] = {}
                feature["frequent-entries"]["most_common_punctuations"][string.punctuation[i]]["count"] = counts_column_punc[i]
                feature["frequent-entries"]["most_common_punctuations"][string.punctuation[i]]["density_of_all"] = counts_column_punc[i] / float(number_of_chars)
                feature["frequent-entries"]["most_common_punctuations"][string.punctuation[i]]["density_of_cell"] = puncs_density_average[i]
                # calculate outlier
                outlier_array = helper_outlier_calcu(cell_density_array[:, i], weight_outlier)
                feature["frequent-entries"]["most_common_punctuations"][string.punctuation[i]]["num_outlier_cells"] = sum(outlier_array)

def helper_outlier_calcu(array, number_of_sigma):
    """
    input: array is a 1D numpy array, number_of_sigma is a integer.
    output: boolean array, size same with input array; true -> is outlier, false -> not outlier 
    outlier def:
        the values that not within mean +- (number_of_sigma * sigma) of the statics of the whole list
    """
    mean = np.mean(array)
    std = np.std(array)
    upper_bound = mean + number_of_sigma * std
    lower_bound = mean - number_of_sigma * std
    outlier = (array > upper_bound) + (array < lower_bound)
    return outlier

