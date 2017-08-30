import pandas as pd
from dsbox.executer.executionhelper import NestedData

class FlattenTable(object):
    def joinedTable(self, df):
        df = df.copy()
        nested = [ isinstance(x, NestedData) for x in df.iloc[0] ]
        index_col = nested.index(True)
        nested_data_column = df.iloc[:,index_col]
        nested_table = nested_data_column[0].nested_data
        index_col_name = nested_data_column[0].index_column
        indices = [x.index for x in nested_data_column]
        df[index_col_name] = indices
        # varun: Merge, but don't change the index 
        joined = pd.merge(df, nested_table, on=index_col_name)
        #joined = df.set_index(index_col_name).join(nested_table.set_index(index_col_name))
        return joined

    def fit(self, df, label=None):
        nested = [isinstance(x, NestedData) for x in df.iloc[0] ]
        if not True in nested:
            return df
        return self.joinedTable(df)

    def transform(self, df, label=None):
        nested = [isinstance(x, NestedData) for x in df.iloc[0] ]
        if not True in nested:
            return df
        return self.joinedTable(df)

    def fit_transform(self, df, label=None):
        nested = [isinstance(x, NestedData) for x in df.iloc[0] ]
        if not True in nested:
            return df
        return self.joinedTable(df)
