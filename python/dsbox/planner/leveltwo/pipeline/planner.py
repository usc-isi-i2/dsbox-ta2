import json
import pandas as pd

from L1_planner import L1_Planner
from L2_planner import L2_Planner
from profiler.data_profiler import Profiler
from schema.profile import Profile

class Planner(object):
    """
    This is the overall "planning" coordinator. It is passed in the problem directory
    and the primitives library directory, and it generates plans by calling out to L1, L2
    and L3 planners.
    """
    def __init__(self, directory, libdir):
        self.directory = directory
        self.problem = self._load_json(directory + "/problemSchema.json")
        self.task_type = self.problem['taskType']
        self.task_subtype = self.problem.get('taskSubType', None)
        self.metric = self._convert_metric(self.problem.get('metric'))
                
        self.schema = self._load_json(directory + "/data/dataSchema.json")
        self.columns = self.schema['trainData']['trainData']        
        self.targets = self.schema['trainData']['trainTargets']
                
        self.train_data = self._read_data(directory +'/data/trainData.csv', 
                                         self.columns)
        self.train_labels = self._read_data(directory +'/data/trainTargets.csv', 
                                           self.targets)

        self.l1_planner = L1_Planner(libdir)
        self.l2_planner = L2_Planner(libdir)
        self.plan_list = []
    
    
    def start(self):
        print("Task type: %s" % self.task_type)
        print("Metric: %s" % self.metric)
        
        # Get data details
        df = pd.DataFrame(self.train_data, columns = self.train_data.columns)
        df_lbl = pd.DataFrame(self.train_labels, columns = self.train_labels.columns)
        
        df_profile = self._get_data_profile(df)
        print("Data profile: %s" % df_profile)

        l1_pipelines = self.l1_planner.get_pipelines(self.task_type, self.task_subtype, df)
        print "\nL1 Pipelines:\n-------------"
        print(l1_pipelines)
        print("-------------")
                        
        l2_pipelines = []
        for l1_pipeline in l1_pipelines:
            l2_pipeline_list = self.l2_planner.expand_pipeline(l1_pipeline, df_profile)
            if l2_pipeline_list:
                l2_pipelines.extend(l2_pipeline_list)

        print "\nL2 Pipelines:\n-------------"
        print(l2_pipelines)

        l2_exec_pipelines = []
        for l2_pipeline in l2_pipelines:
            expipe = self.l2_planner.patch_and_execute_pipeline(
                    l2_pipeline, df, df_lbl, self.columns, self.metric)
            if expipe:
                l2_exec_pipelines.append(expipe)
        
        l2_exec_pipelines = sorted(l2_exec_pipelines, key=lambda x: x[1])

        print "\nL2 Executed Pipelines:\n-------------"
        print(l2_exec_pipelines)
    
    def _load_json(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d

    def _read_data(self, csvfile, cols):
        df = pd.read_csv(csvfile)
        for col in cols:
            colname = col['varName']
            if col['varRole'] == 'file':            
                for index, row in df.iterrows():
                    filename = row[colname]
                    with open(self.directory + '/data/raw_data/' + filename, 'r') as myfile:
                        txt = myfile.read()
                        df.set_value(index, colname, txt)

        return df
        
    def _convert_metric(self, metric):
        metric = metric.lower()
        if metric != "f1" and metric[0:2] == "f1":
            metric = "f1_" + metric[2:]
        elif metric == "meansquarederror":
            metric = "neg_mean_squared_error" # FIXME
        elif metric == "rootmeansquarederror":
            metric = "neg_mean_squared_error" # FIXME
        return metric
        
    def _get_data_profile(self, df):
        df_profile_raw = Profiler(df)
        return Profile(df_profile_raw)
    

    def stop(self):
        '''
        Stop planning, and write out the current list (sorted by metric)
        '''
