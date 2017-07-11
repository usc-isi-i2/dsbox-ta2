import json
import pandas as pd

from L1_planner import L1_Planner
from L2_planner import L2_Planner
from profiler.data_profiler import Profiler
from schema.profile import Profile

class Planner(object):
    '''
    The actual planning algorithm
    '''
    def __init__(self, directory, libdir):
        self.directory = directory
        self.problem = self.loadjson(directory + "/problemSchema.json")
        self.schema = self.loadjson(directory + "/data/dataSchema.json")
        self.train_data = pd.read_csv(directory +'/data/trainData.csv')
        self.train_labels = pd.read_csv(directory +'/data/trainTargets.csv')
        self.task_type = self.problem['taskType']
        self.task_subtype = self.problem.get('taskSubType', None)
        self.targets = self.schema['trainData']['trainTargets']
        self.columns = self.schema['trainData']['trainData']
        self.l1_planner = L1_Planner(libdir)
        self.l2_planner = L2_Planner(libdir)
        self.plan_list = []

    def loadjson(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d

    def start(self):
        print("Task type: %s" % self.task_type)
        
        # Get data details
        indexcol = self.get_index_column()
        df = pd.DataFrame(self.train_data, columns = self.train_data.columns)
        df_lbl = pd.DataFrame(self.train_labels, columns = self.train_labels.columns)
        
        df_profile = self.get_data_profile(df)
        print("Data profile: %s" % df_profile)

        l1_pipelines = self.l1_planner.get_pipelines(self.task_type, df)
        print "\nL1 Pipelines:\n-------------"
        print(l1_pipelines)
        print("-------------")
        
        l2_pipelines = []
        for l1_pipeline in l1_pipelines:
            l2_pipelines.extend(self.l2_planner.expand_pipeline(l1_pipeline, df_profile))

        l2_exec_pipelines = []
        for l2_pipeline in l2_pipelines:
            l2_exec_pipelines.append(
                self.l2_planner.patch_and_execute_pipeline(
                    l2_pipeline, df, df_lbl, indexcol)) 
        
        print "\nL2 Pipelines:\n-------------"
        print(l2_exec_pipelines)
        
    def get_data_profile(self, df):
        df_profile_raw = Profiler(df)
        return Profile(df_profile_raw)
            
        '''
        # Choose algorithm
        #algo = linear_model.LogisticRegression()
        algos = [
            ('Logistic Regression', linear_model.LogisticRegression()),
            ('KNN', neighbors.KNeighborsClassifier()),
            ('Naive Bayes', naive_bayes.MultinomialNB())
        ]
        
        # Add glue components (Make numeric if needed, Remove missing values if needed)
        df = self.add_glue(df)
        
        # Split Training and Test Data
        train, test = self.get_train_test(df, indexcol)
        train_lbl, test_lbl = self.get_train_test(df_lbl, indexcol)     

        print (train_lbl)
        print (test_lbl)
        for algo in algos:
            print('%s Score: %f' % (algo[0],
                  algo[1].fit(train, train_lbl).score(test, test_lbl)))

        '''

    def get_index_column(self):
        for col in self.columns:
            if col['varRole'] == 'index':
                return col['varName']
        return None
    

    def stop(self):
        '''
        Stop planning, and write out the current list (sorted by metric)
        '''
