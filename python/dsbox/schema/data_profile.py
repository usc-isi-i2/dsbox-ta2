from dsbox.profiler.data.data_profiler import DataProfiler
from dsbox.schema.profile_schema import DataProfileType as dpt

class DataProfile(object):
    """
    This class holds the profile for either the whole data, or a column
    """

    def __init__(self, dataframe):
        self.profiler_data = DataProfiler(dataframe)
        self.profile = self.getDefaultProfile()
        self.columns = {}
        for column_name, col_data in self.profiler_data.result.items():
            profile = self.parseColumnProfile(col_data)
            self.addColumnProfile(column_name, profile)

    def addColumnProfile(self, column, profile):
        self.columns[column] = profile
        #print "Column %s: %s" % (column, profile)
        if not profile.get(dpt.NUMERICAL):
            self.profile[dpt.NUMERICAL] = False
        if profile.get(dpt.MISSING_VALUES):
            self.profile[dpt.MISSING_VALUES] = True
        if profile.get(dpt.UNIQUE):
            self.profile[dpt.UNIQUE] = True
        if profile.get(dpt.NEGATIVE):
            self.profile[dpt.NEGATIVE] = True
        if profile.get(dpt.NESTED_DATA):
            self.profile[dpt.NESTED_DATA] = True

    def getDefaultProfile(self):
        # By default, we mark profile as
        # - numerical,
        # - no missing values
        # - not unique
        # - not negative
        return {
            dpt.NUMERICAL : True,
            dpt.MISSING_VALUES: False,
            dpt.UNIQUE : False,
            dpt.NEGATIVE : False,
            dpt.NESTED_DATA : False
        }

    def getProfile(self):
        return self.profile

    def getColumnProfile(self, column):
        return self.columns[column]

    # TODO: Check for UNIQUE
    def parseColumnProfile(self, col_data):
        profile = self.getDefaultProfile()
        profile[dpt.NUMERICAL] = False
        if col_data['missing'].get('num_missing', 0) > 0:
            profile[dpt.MISSING_VALUES] = True
        if col_data.get('numeric_stats'):
            profile[dpt.NUMERICAL] = True
            numneg = col_data.get('numeric_stats').get('num_negative')
            if numneg > 0:
                profile[dpt.NEGATIVE] = True
        if ('special_type' in col_data
            and 'data_type' in col_data['special_type']
            and col_data['special_type']['data_type'] == 'Nested Data'):
            profile[dpt.NESTED_DATA] = True
        return profile

    def __str__(self):
        return "%s" % self.profile

    def __repr__(self):
        return "%s" % self.profile
