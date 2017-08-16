import os
import json
import shutil
import os.path
import pandas as pd
from sklearn.externals import joblib
from dsbox.schema import VariableFileType

class ExecutionHelper(object):
    def __init__(self, directory, outputdir, csvfile=None):
        self.directory = os.path.abspath(directory)
        self.outputdir = os.path.abspath(outputdir)
        self.schema = self.load_json(directory + "/data/dataSchema.json")
        self.columns = self.schema['trainData']['trainData']
        self.targets = self.schema['trainData']['trainTargets']
        self.indexcol = self.get_index_column(self.columns)
        self.media_type = self.get_media_type(self.schema)
        self.data = None
        if csvfile:
            self.data = self.read_data(directory + "/data/" + csvfile, self.columns, self.indexcol)


    def get_index_column(self, columns):
        for col in columns:
            if col['varRole'] == 'index':
                return col['varName']
        return None

    def load_json(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d

    def read_data(self, csvfile, cols, indexcol):
        if not os.path.exists(csvfile) and csvfile.endswith('.gz'):
            csvfile = csvfile[:-3]
        df = pd.read_csv(csvfile, index_col=indexcol)
        #df = df.reindex(pd.RangeIndex(df.index.max()+1)).ffill()
        df = df.reset_index(drop=True)
        for col in cols:
            colname = col['varName']
            if col['varRole'] == 'file':
             for index, row in df.iterrows():
                filename = row[colname]
                with open(self.directory + '/data/raw_data/' + filename, 'rb') as myfile:
                    txt = myfile.read()
                    df.set_value(index, colname, txt)
        return df

    def get_media_type(self, schema):
        if schema.get('rawData', False):
            types = schema.get('rawDataFileTypes', {})
            for ext in types.keys():
                typ = types[ext]
                if typ.startswith("text/csv"):
                    return VariableFileType.TABULAR
                elif typ.startswith("text"):
                    return VariableFileType.TEXT
                elif typ.startswith("image"):
                    return VariableFileType.IMAGE
                elif typ.startswith("audio"):
                    return VariableFileType.AUDIO
                elif typ.startswith("video"):
                    return VariableFileType.VIDEO
        return None

    def columns_of_role(self, columns, role):
        cols = []
        for c in columns:
            if c['varRole'] == role:
                cols.append(c['varName'])
        return cols

    def featurise(self, df, primex):
        ncols = [col.format() for col in df.columns]
        featurecols = self.columns_of_role(self.columns, "file")
        for col in featurecols:
            primex.fit(df[col])
            nvals = primex.transform(df[col])
            fcols = [(col.format() + "_" + feature) 
                    for feature in primex.get_feature_names()]
            newdf = pd.DataFrame(nvals.toarray(), columns=fcols)
            del df[col]
            ncols = ncols + fcols
            ncols.remove(col)
            df = pd.concat([df, newdf], axis=1)
            df.columns=ncols
        return df

    def create_pipeline_executable(self, pipeline, pipeid):
        imports = ["sys", "sklearn.externals", "pandas"]
        statements = [
                "from os import path", 
                "from helper import ExecutionHelper", 
                "sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))",
                "", 
                "# Pipeline : %s" % str(pipeline), 
                ""]
        statements.append("hp = ExecutionHelper('%s', '.', 'testData.csv.gz')" % self.directory)
        statements.append("testdata = hp.data")
        #statements.append("testdata = pandas.read_csv('%s/data/testData.csv.gz', index_col='%s')" % (self.directory, self.indexcol))
        index = 1
        for primitive in pipeline:
            primid = "primitive_%s" % str(index)
            
            # Initialize primitive
            if primitive.is_persistent:
                primfile = "%s/models/%s.%s.pkl" % (self.outputdir, pipeid, primid)
                statements.append("%s = sklearn.externals.joblib.load('%s')" % (primid, primfile))
                joblib.dump(primitive.executable, primfile)
            else:
                mod, cls = primitive.cls.rsplit('.', 1)
                imports.append(mod)

                args = []
                kwargs = {}
                if primitive.getInitKeywordArgs():
                    kwargs = self.l2_planner._process_kwargs(primitive.getInitKwargs(), self.task_type, self.metric)
                if primitive.getInitArgs():
                    args = self.l2_planner._process_args(primitive.getInitArgs(), self.task_type, self.metric)
                statements.append("args = %s" % args)
                statements.append("kwargs = %s" % kwargs)
                statements.append("%s = %s(*args, **kwargs)" % (primid, primitive.cls))
            
            if primitive.task == "Modeling":
                statements.append("print %s.predict(testdata)" % primid)
            else:
                if primitive.task == "PreProcessing":
                    if primitive.column_primitive:
                        statements.append("for col in testdata.columns:")
                        statements.append("    testdata[col] = %s.fit_transform(testdata[col])" % primid)
                    else:
                        statements.append("testdata = %s.fit_transform(testdata)" % primid)
            
                elif primitive.task == "FeatureExtraction":
                    statements.append("testdata = hp.featurise(testdata, %s)" % primid)

            index += 1
    
        # Write executable
        with open("%s/executables/%s.py" % (self.outputdir, pipeid), 'a') as exfile:
            for imp in set(imports):
                exfile.write("import %s\n" % imp)
            for st in statements:
                exfile.write(st+"\n")

        # Copy helper.py file (this file)
        helperfile = "%s/executables/helper.py" % self.outputdir
        if not os.path.isfile(helperfile):
            thisfile = os.path.abspath(__file__)
            thisfile = thisfile.replace(".pyc", ".py")
            shutil.copyfile(thisfile, helperfile)

