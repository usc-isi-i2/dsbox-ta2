from constants import ProfileConstants as pc

class Profile(object):
    """
    This class holds the profile for either the whole data, or a column
    """
    
    def __init__(self, profiler_data):
        self.profile = self.getDefaultProfile()
        self.columns = {}
        for column_name, col_data in profiler_data.result.items():
            profile = self.parseColumnProfile(col_data)
            self.addColumnProfile(column_name, profile)

    def addColumnProfile(self, column, profile):
        self.columns[column] = profile
        print "Column %s: %s" % (column, profile)
        if not profile.get(pc.NUMERICAL):
            self.profile[pc.NUMERICAL] = False
        if profile.get(pc.MISSING_VALUES):
            self.profile[pc.MISSING_VALUES] = True
        if profile.get(pc.UNIQUE):
            self.profile[pc.UNIQUE] = True
        if profile.get(pc.NEGATIVE):
            self.profile[pc.NEGATIVE] = True      
    
    def getDefaultProfile(self):
        # By default, we mark profile as 
        # - numerical,
        # - no missing values
        # - not unique
        # - not negative        
        return {
            pc.NUMERICAL : True,            
            pc.MISSING_VALUES: False, 
            pc.UNIQUE : False, 
            pc.NEGATIVE : False
        }
             
    def getProfile(self):
        return self.profile
    
    def getColumnProfile(self, column):
        return self.columns[column]
    
    # TODO: Check for UNIQUE
    def parseColumnProfile(self, col_data):
        profile = self.getDefaultProfile()
        profile[pc.NUMERICAL] = False
        if col_data['missing'].get('num_missing', 0) > 0:
            profile[pc.MISSING_VALUES] = True
        if col_data.get('numeric_stats'):
            profile[pc.NUMERICAL] = True
            numneg = col_data.get('numeric_stats').get('num_negative')
            if numneg > 0:
                profile[pc.NEGATIVE] = True
        return profile
    
    def __str__(self):
        return "%s" % self.profile

    def __repr__(self):
        return "%s" % self.profile   