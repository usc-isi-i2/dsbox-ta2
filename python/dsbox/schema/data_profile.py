from dsbox.profiler.data.data_profiler import DataProfiler
from dsbox.schema.profile_schema import DataProfileType as dpt

class DataProfile(object):
    """
    This class holds the profile for either the whole data, or a column
    """

    MINCHARS_FOR_TEXT = 25

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
        if profile.get(dpt.TEXT):
            self.profile[dpt.TEXT] = True
        if profile.get(dpt.LIST):
            self.profile[dpt.LIST] = True

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
            dpt.TEXT : False,
            dpt.LIST : False
        }

    def getProfile(self):
        return self.profile

    def getColumnProfile(self, column):
        return self.columns[column]

    # TODO: Check for UNIQUE
    def parseColumnProfile(self, col_data):
        profile = self.getDefaultProfile()
        profile[dpt.NUMERICAL] = False
        # Mark missing values
        if col_data['missing'].get('num_missing', 0) > 0:
            profile[dpt.MISSING_VALUES] = True

        # Mark Numeric Values
        if col_data.get('numeric_stats', None) is not None:
            stats = col_data.get('numeric_stats').get('integer', None)
            if stats is None:
                stats = col_data.get('numeric_stats').get('decimal', None)
            if stats is not None:
                # If all values numeric, mark as numeric
                num_nonblank = col_data['missing'].get('num_nonblank')
                if stats['count'] == num_nonblank:
                    profile[dpt.NUMERICAL] = True
                    # Mark Negative
                    numneg = col_data.get('numeric_stats').get('num_negative', 0)
                    if numneg > 0:
                        profile[dpt.NEGATIVE] = True

        # Mark List
        if col_data.get('special_type', None) is not None:
            stype = col_data.get('special_type')
            if stype['data_type'] == 'list':
                profile[dpt.LIST] = True

        # Mark Text
        if col_data.get('length', None) is not None:
            stats = col_data.get('length')
            avg = stats['character']['average']
            if avg > DataProfile.MINCHARS_FOR_TEXT:
                profile[dpt.TEXT] = True


        return profile

    def __str__(self):
        return "%s" % self.profile

    def __repr__(self):
        return "%s" % self.profile
