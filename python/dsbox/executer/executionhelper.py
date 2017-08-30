import os
import sys
import json
import copy
import shutil
import os.path
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.model_selection import KFold

from dsbox.schema.dataset_schema import VariableFileType
from dsbox.executer.execution import Execution

import scipy.sparse.csr

import stopit
import inspect
import importlib
import traceback

REMOTE = False

class ExecutionHelper(object):
    def __init__(self, directory, outputdir, csvfile=None):
        self.e = Execution()
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


    def instantiate_primitive(self, primitive, args, kwargs):
        if REMOTE:
            return self.e.execute(primitive.cls, args=args, kwargs=kwargs)
        mod, cls = primitive.cls.rsplit('.', 1)
        try:
            module = importlib.import_module(mod)
            PrimitiveClass = getattr(module, cls)
            return PrimitiveClass(*args, **kwargs)
        except Exception as e:
            sys.stderr.write("ERROR _instantiate_primitive(%s) : %s\n" % (primitive, e))
            traceback.print_exc()
            return None

    @stopit.threading_timeoutable()
    def execute_primitive(self, primitive, df, df_lbl, cur_profile=None):
        # If this is a non-modeling primitive, fit & transform
        if primitive.column_primitive:
            for col in df.columns:
                colprofile = cur_profile.columns[col]
                if self._profile_matches_precondition(primitive.preconditions, colprofile):
                    try:
                        df[col] = self._execute_primitive(primitive, df[col])
                    except Exception as e:
                        sys.stderr.write("ERROR execute_primitive(%s): %s\n" % (primitive, e))
                        traceback.print_exc()
                        return None
        else:
            if self._profile_matches_precondition(primitive.preconditions, profile.profile):
                df = self._execute_primitive(primitive, df, df_lbl)
        return df

    def _profile_matches_precondition(self, preconditions, profile):
        for precondition in preconditions.keys():
            prec_value = preconditions[precondition]
            if prec_value != profile.get(precondition, None):
                return False
        return True

    def _execute_primitive(self, primitive, df, df_lbl=None):
        args = [df]
        if df_lbl is not None:
            args.append(df_lbl)

        if primitive.is_persistent:
            if REMOTE:
                primitive.executable = self.e.execute('fit', args=args, kwargs=None, obj=primitive.executable, objreturn=True)
                return self.e.execute('transform', args=args, kwargs=None, obj=primitive.executable)
            else:
                primitive.executable.fit(*args)
                return primitive.executable.transform(*args)
        else:
            if REMOTE:
                return self.e.execute('fit_transform', args=args, kwargs=None, obj=primitive.executable)
            else:
                return primitive.executable.fit_transform(*args)


    @stopit.threading_timeoutable()
    def cross_validation_score(self, prim, X, y, metric, metric_function, cv=4):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        vals = []
        for k, (train, test) in enumerate(kf.split(X, y)):
            if REMOTE:
                prim = self.e.execute('fit', args=[X.take(train, axis=0), y.take(train, axis=0)], kwargs=None, obj=prim, objreturn=True)
                ypred = self.e.execute('predict', args=[X.take(test, axis=0)], kwargs=None, obj=prim)
            else:
                prim.fit(X.take(train, axis=0), y.take(train, axis=0))
                ypred = prim.predict(X.take(test, axis=0))
            val = self._call_function(metric_function, y.take(test, axis=0), ypred)
            vals.append(val)

        # fit the model finally over the whole training data for evaluation later over actual test data
        if REMOTE:
            prim = self.e.execute('fit', args=[X, y], kwargs=None, obj=prim, objreturn=True)
        else:
            prim.fit(X, y)
        return np.average(vals)

    def _as_tensor(self, image_list):
        from keras.preprocessing import image
        shape = (len(image_list), ) + image.img_to_array(image_list[0]).shape
        result = np.empty(shape)
        for i in range(len(image_list)):
            result[i] = image.img_to_array(image_list[i])
        return result

    @stopit.threading_timeoutable()
    def featurise(self, df, primex):
        ncols = [col.format() for col in df.columns]
        #primeorig = copy.deepcopy(primex)
        featurecols = self.columns_of_role(self.columns, "file")
        for col in featurecols:
            if self.media_type == VariableFileType.TEXT:
                # Using an unfitted primitive for each column (needed for Corex)
                #primex = primeorig
                #df_col = pd.DataFrame(df[col])
                #primex.fit(df_col)
                nvals = primex.fit_transform(df[col])
                if isinstance(nvals, scipy.sparse.csr.csr_matrix):
                    nvals = nvals.todense()
                #fcols = [(col.format() + "_" + feature) for feature in primex.get_feature_names()]
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            elif self.media_type == VariableFileType.IMAGE:
                image_tensor = self._as_tensor(df[col].values)
                nvals = primex.transform(image_tensor)
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
        return df

    def _call_function(self, scoring_function, *args):
        mod = inspect.getmodule(scoring_function)
        try:
            module = importlib.import_module(mod.__name__)
            return scoring_function(*args)
        except Exception as e:
            sys.stderr.write("ERROR _call_function %s: %s\n" % (scoring_function, e))
            traceback.print_exc()
            return None

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
                filepath = self.directory + '/data/raw_data/' + row[colname]
                if self.media_type == VariableFileType.TEXT:
                    with open(filepath, 'rb') as myfile:
                        txt = myfile.read()
                        df.set_value(index, colname, txt)
                elif self.media_type == VariableFileType.IMAGE:
                    from keras.preprocessing import image
                    df.set_value(index, colname, image.load_img(filepath, target_size=(224, 224)))
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

    def create_pipeline_executable(self, pipeline, pipeid):
        imports = ["sys", "sklearn.externals", "pandas"]
        statements = [
                "from os import path",
                "from dsbox.executer.executionhelper import ExecutionHelper",
                #"sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))",
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
                try:
                    primfile = "%s/models/%s.%s.pkl" % (self.outputdir, pipeid, primid)
                    statements.append("%s = sklearn.externals.joblib.load('%s')" % (primid, primfile))
                    joblib.dump(primitive.executable, primfile)
                except Exception as e:
                    sys.stderr.write("ERROR pickling %s : %s\n" % (primitive.name, e))
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

        libdir = self.outputdir + "/executables/dsbox"
        schemadir = libdir + "/schema"
        execdir = libdir + "/executer"
        if not os.path.exists(libdir):
            os.makedirs(libdir)
            self._init_initfile(libdir)
        if not os.path.exists(schemadir):
            os.makedirs(schemadir)
            self._init_initfile(schemadir)
        if not os.path.exists(execdir):
            os.makedirs(execdir)
            self._init_initfile(execdir)

        # Copy executionhelper.py, execution.py & dataset_schema.py file (this file)
        exfile = "%s/executables/dsbox/executer/executionhelper.py" % self.outputdir
        if not os.path.isfile(exfile):
            thisfile = os.path.abspath(__file__)
            thisfile = thisfile.replace(".pyc", ".py")
            shutil.copyfile(thisfile, exfile)
        exfile = "%s/executables/dsbox/executer/execution.py" % self.outputdir
        if not os.path.isfile(exfile):
            origfile = os.path.dirname(os.path.abspath(__file__)) + "/execution.py"
            origfile = origfile.replace(".pyc", ".py")
            shutil.copyfile(origfile, exfile)
        schfile = "%s/executables/dsbox/schema/dataset_schema.py" % self.outputdir
        if not os.path.isfile(schfile):
            origfile = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/schema/dataset_schema.py"
            origfile = origfile.replace(".pyc", ".py")
            shutil.copyfile(origfile, schfile)


    def _init_initfile(self, dir):
        with open(dir+"/__init__.py", "w") as init_file:
            init_file.write("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
