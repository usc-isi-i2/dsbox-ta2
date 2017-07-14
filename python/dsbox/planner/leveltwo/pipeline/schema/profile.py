class Profile(object):
    """
    This class holds the profile for either the whole data, or a column
    """
    MISSING_VALUES = 0b00000001
    NUMERICAL = 0b00000010
    NON_NEGATIVE = 0b00000100
    
    # Inverse of the above profiles (allowing 16 profiles currently)
    NO_MISSING_VALUES = MISSING_VALUES << 16
    NO_NUMERICAL = NUMERICAL << 16
    NO_NON_NEGATIVE = NON_NEGATIVE << 16
    
    def __init__(self, profiler_data):
        # By default, we mark profile as numerical and with no missing data
        self.profile = Profile.NO_MISSING_VALUES | Profile.NUMERICAL
        self.columns = {}
        for column_name, col_data in profiler_data.result.items():
            profile = self.parseColumnProfile(col_data)
            self.addColumnProfile(column_name, profile)

    def addColumnProfile(self, column, profile):
        self.columns[column] = profile
        # If even one of the columns has missing values,
        # then mark the data as having missing values
        if profile & Profile.MISSING_VALUES:
            #print "Column %s has missing values" % column
            self.profile |= Profile.MISSING_VALUES
            self.profile &= ~Profile.NO_MISSING_VALUES
        # If even one of the columns is not numerical, 
        # then mark the data as non-numerical
        if not (profile & Profile.NUMERICAL):
            #print "Column %s is not numerical" % column
            self.profile |= Profile.NO_NUMERICAL
            self.profile &= ~Profile.NUMERICAL

    
    def getProfile(self):
        return self.profile
    
    def getColumnProfile(self, column):
        return self.columns[column]
    
    def parseColumnProfile(self, col_data):
        profile = 0
        if col_data['missing'].get('num_missing', 0) > 0:
            profile |= Profile.MISSING_VALUES
            profile &= ~Profile.NO_MISSING_VALUES
        if col_data.get('numeric_stats'):
            profile |= Profile.NUMERICAL
            profile &= ~Profile.NO_NUMERICAL
        return profile
    
    def __str__(self):
        return self.toString(self.profile)
    
    def toString(self, pbit):
        txts = []
        #txts.append(bin(pbit))
        if pbit & Profile.NO_MISSING_VALUES:
            txts.append("No Missing Values")
        if pbit & Profile.MISSING_VALUES:
            txts.append("Missing Values")
        if pbit & Profile.NO_NUMERICAL:
            txts.append("Not Numerical")
        if pbit & Profile.NUMERICAL:
            txts.append("Numerical")
        return ", ".join(txts)
        