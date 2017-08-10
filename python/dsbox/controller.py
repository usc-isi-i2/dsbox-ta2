import os
import os.path
import json
import uuid
import shutil
import pandas as pd

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.schema.data_profile import DataProfile
from dsbox.schema import TaskType, TaskSubType

from sklearn.externals import joblib

class Controller(object):
    """
    This is the overall "planning" coordinator. It is passed in the problem directory
    and the primitives library directory, and it generates plans by calling out to L1, L2
    and L3 planners.
    """
    def __init__(self, directory, libdir, outputdir):
        self.directory = os.path.abspath(directory)
        self.problem = self._load_json(directory + "/problemSchema.json")
        self.task_type = TaskType(self.problem['taskType'])
        self.task_subtype = None
        subtype = self.problem.get('taskSubType', None)
        if subtype:
            subtype = subtype.replace(self.task_type.value.title(), "")
            self.task_subtype = TaskSubType(subtype)
        self.metric = self._convert_metric(self.problem.get('metric'))
                
        self.schema = self._load_json(directory + "/data/dataSchema.json")
        self.targets = self.schema['trainData']['trainTargets']
        self.columns = self.schema['trainData']['trainData']        
        self.indexcol = self._get_index_column(self.columns)
                
        self.train_data = self._read_data(directory +'/data/trainData.csv.gz', 
                                         self.columns, self.indexcol)
        self.train_labels = self._read_data(directory +'/data/trainTargets.csv.gz', 
                                           self.targets, self.indexcol)

        self.libdir = os.path.abspath(libdir)
        self.outputdir = os.path.abspath(outputdir)
        if os.path.exists(self.outputdir):
            shutil.rmtree(self.outputdir)
        os.makedirs(self.outputdir+"/models")
        os.makedirs(self.outputdir+"/executables")

        self.logfile = open("%s/log.txt" % self.outputdir, 'w')
        self.pipelinesfile = open("%s/pipelines.txt" % self.outputdir, 'w')

        self.l1_planner = LevelOnePlannerProxy(self.libdir, self.task_type, self.task_subtype)
        self.l2_planner = LevelTwoPlanner(self.libdir)

        self.plan_list = []
    
    
    def convert_l1_to_l2(self, pipeline):
        pipeline.get_primitives()

    def start(self, cutoff=5):
        self.logfile.write("Task type: %s\n" % self.task_type)
        self.logfile.write("Metric: %s\n" % self.metric)
        
        # Get data details
        df = pd.DataFrame(self.train_data, columns = self.train_data.columns)
        df_lbl = pd.DataFrame(self.train_labels, columns = self.train_labels.columns)
        
        df_profile = self._get_data_profile(df)
        self.logfile.write("Data profile: %s\n" % df_profile)

        l1_pipelines_handled = {}
        l1_pipelines = self.l1_planner.get_pipelines(df)
        l2_exec_pipelines = []

        while len(l1_pipelines) > 0:
            self.logfile.write("\nL1 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l1_pipelines))
            self.logfile.write("-------------\n")
                            
            l2_l1_map = {}

            l2_pipelines = []
            for l1_pipeline in l1_pipelines:
                l2_pipeline_list = self.l2_planner.expand_pipeline(l1_pipeline, df_profile)
                l1_pipelines_handled[str(l1_pipeline)] = True
                if l2_pipeline_list:
                    for l2_pipeline in l2_pipeline_list:
                        l2_l1_map[str(l2_pipeline)] = l1_pipeline
                        l2_pipelines.append(l2_pipeline)

            self.logfile.write("\nL2 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_pipelines))

            for l2_pipeline in l2_pipelines:
                expipe = self.l2_planner.patch_and_execute_pipeline(
                        l2_pipeline, df, df_lbl, self.columns, self.metric)
                if expipe:
                    l2_exec_pipelines.append(expipe)
            
            l2_exec_pipelines = sorted(l2_exec_pipelines, key=lambda x: -x[1])
            self.logfile.write("\nL2 Executed Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_exec_pipelines))

            # TODO: Do Pipeline Hyperparameter Tuning

            # Pick top N pipelines, and get similar pipelines to it from the L1 planner to further explore
            l1_related_pipelines = []
            for index in range(0, cutoff):
                if index >= len(l2_exec_pipelines):
                    break
                pipeline = l2_l1_map.get(str(l2_exec_pipelines[index][0]))
                if pipeline:
                    related_pipelines = self.l1_planner.get_related_pipelines(pipeline)
                    for related_pipeline in related_pipelines:
                        if not l1_pipelines_handled.get(str(related_pipeline), False):
                            l1_related_pipelines.append(related_pipeline)

        
            self.logfile.write("\nRelated L1 Pipelines to top %d L2 Pipelines:\n-------------\n" % cutoff)
            self.logfile.write("%s\n" % str(l1_related_pipelines))
            l1_pipelines = l1_related_pipelines

        # Ended planners
        
        # Create executables
        self.pipelinesfile.write("# Pipelines ranked by metric (%s)\n" % self.metric)
        for index in range(0, len(l2_exec_pipelines)):
            pipeline = l2_exec_pipelines[index][0]
            self.pipelinesfile.write("%s : %2.4f\n" % (pipeline, l2_exec_pipelines[index][1]))
            pipeline_name = str(index+1) + "." + str(uuid.uuid1())
            self._create_pipeline_executable(pipeline, pipeline_name)

    def _create_pipeline_executable(self, pipeline, pipeid):
        imports = ["sklearn.externals", "pandas"]
        statements = ["", "# Pipeline : %s" % str(pipeline), ""]
        statements.append("testdata = pandas.read_csv('%s/data/testData.csv.gz', index_col='%s')" % (self.directory, self.indexcol))
        index = 1
        for primitive in pipeline:
            primid = "primitive_%s" % str(index)
            mod, cls = primitive.cls.rsplit('.', 1)
            imports.append(mod)
            if primitive.task == "Modeling":
                statements.append("%s = sklearn.externals.joblib.load('%s/models/%s.pkl')" % (primid, self.outputdir, pipeid))
                joblib.dump(primitive.executable, "%s/models/%s.pkl" % (self.outputdir, pipeid))
                statements.append("print %s.predict(testdata)" % primid)
            if primitive.task == "PreProcessing":
                statements.append("%s = %s()" % (primid, primitive.cls))
                if primitive.column_primitive:
                    statements.append("for col in testdata.columns:")
                    statements.append("    testdata[col] = %s.fit_transform(testdata[col])" % primid)
                else:
                    statements.append("testdata = %s.fit_transform(testdata)" % primid)
            index += 1

        with open("%s/executables/%s.py" % (self.outputdir, pipeid), 'a') as exfile:
            for imp in set(imports):
                exfile.write("import %s\n" % imp)
            for st in statements:
                exfile.write(st+"\n")

    def _load_json(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d

    def _read_data(self, csvfile, cols, indexcol):
        df = pd.read_csv(csvfile, index_col=indexcol)
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
        return DataProfile(df)
    
    def _get_index_column(self, columns):
        for col in columns:
            if col['varRole'] == 'index':
                return col['varName']
        return None

    def stop(self):
        '''
        Stop planning, and write out the current list (sorted by metric)
        '''
