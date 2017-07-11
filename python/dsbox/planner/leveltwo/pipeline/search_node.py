class SearchNode (object):
    '''
    A node in the search
    '''
    def __init__(self, parent, plan):
        self.parent = parent
        self.plan = plan