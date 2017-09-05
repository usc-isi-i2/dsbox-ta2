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

from builtins import range

REMOTE = False

class ExecutionHelper(object):
    def __init__(self, data_directory, outputdir, csvfile=None):
        self.e = Execution()
        self.directory = os.path.abspath(data_directory)
        self.outputdir = os.path.abspath(outputdir)
        self.schema = self.load_json(self.directory + "/dataSchema.json")
        self.columns = self.schema['trainData']['trainData']
        self.targets = self.schema['trainData']['trainTargets']
        self.indexcol = self.get_index_column(self.columns)
        self.media_type = self.get_media_type(self.schema)
        self.data = None
        self.nested_table = dict()
        if csvfile:
            self.data = self.read_data(self.directory + "/" + csvfile, self.columns, self.indexcol)


    def instantiate_primitive(self, primitive):
        if REMOTE:
            return self.e.execute(primitive.cls, args=primitive.init_args, kwargs=primitive.init_kwargs)
        mod, cls = primitive.cls.rsplit('.', 1)
        try:
            module = importlib.import_module(mod)
            PrimitiveClass = getattr(module, cls)
            return PrimitiveClass(*primitive.init_args, **primitive.init_kwargs)
        except Exception as e:
            sys.stderr.write("ERROR _instantiate_primitive(%s) : %s\n" % (primitive, e))
            traceback.print_exc()
            return None

    @stopit.threading_timeoutable()
    def execute_primitive(self, primitive, df, df_lbl, cur_profile=None):
        if primitive.column_primitive:
            primitive.executables = {}
            for col in df.columns:
                executable = self.instantiate_primitive(primitive)
                primitive.executables[col.format()] = executable
                colprofile = cur_profile.columns[col]
                if self._profile_matches_precondition(primitive.preconditions, colprofile):
                    try:
                        df[col] = self._execute_primitive(primitive, executable, df[col])
                    except Exception as e:
                        sys.stderr.write("ERROR execute_primitive(%s): %s\n" % (primitive, e))
                        traceback.print_exc()
                        return None
        else:
            primitive.executables = self.instantiate_primitive(primitive)
            if self._profile_matches_precondition(primitive.preconditions, cur_profile.profile):
                df = self._execute_primitive(primitive, primitive.executables, df, df_lbl)

        return pd.DataFrame(df)

    def _profile_matches_precondition(self, preconditions, profile):
        for precondition in preconditions.keys():
            prec_value = preconditions[precondition]
            if prec_value != profile.get(precondition, None):
                return False
        return True

    def _execute_primitive(self, primitive, executable, df, df_lbl=None):
        args = [df]
        if df_lbl is not None:
            args.append(df_lbl.values.ravel())
        if primitive.is_persistent:
            if REMOTE:
                executable = self.e.execute('fit', args=args, kwargs=None, obj=executable, objreturn=True)
                return self.e.execute('transform', args=args, kwargs=None, obj=executable)
            else:
                executable.fit(*args)
                return executable.transform(*args)
        else:
            if REMOTE:
                return self.e.execute('fit_transform', args=args, kwargs=None, obj=executable)
            else:
                return executable.fit_transform(*args)


    @stopit.threading_timeoutable()
    def cross_validation_score(self, primitive, X, y, metric, metric_function, cv=4):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        vals = []
        for k, (train, test) in enumerate(kf.split(X, y)):
            executable = self.instantiate_primitive(primitive)
            if REMOTE:
                prim = self.e.execute('fit', args=[X.take(train, axis=0), y.take(train, axis=0)], kwargs=None, obj=executable, objreturn=True)
                ypred = self.e.execute('predict', args=[X.take(test, axis=0)], kwargs=None, obj=executable)
            else:
                executable.fit(X.take(train, axis=0), y.take(train, axis=0))
                ypred = executable.predict(X.take(test, axis=0))
            val = self._call_function(metric_function, y.take(test, axis=0), ypred)
            vals.append(val)

        # fit the model finally over the whole training data for evaluation later over actual test data
        executable = self.instantiate_primitive(primitive)
        if REMOTE:
            executable = self.e.execute('fit', args=[X, y], kwargs=None, obj=executable, objreturn=True)
        else:
            executable.fit(X, y)
        primitive.executables = executable
        return np.average(vals)

    def _as_tensor(self, image_list):
        from keras.preprocessing import image
        shape = (len(image_list), ) + image.img_to_array(image_list[0]).shape
        result = np.empty(shape)
        for i in range(len(image_list)):
            result[i] = image.img_to_array(image_list[i])
        return result

    @stopit.threading_timeoutable()
    def featurise(self, df, primitive, testing=False, persistent=False):
        if not (testing and persistent):
            primitive.executables = {}
        ncols = [col.format() for col in df.columns]
        featurecols = self.columns_of_role(self.columns, "file")
        for col in featurecols:
            executable = None
            if not (testing and persistent):
                executable = self.instantiate_primitive(primitive)
            else:
                executable = primitive.executables[col]

            if self.media_type == VariableFileType.TEXT:
                # Using an unfitted primitive for each column (needed for Corex)
                #df_col = pd.DataFrame(df[col])
                #executable.fit(df_col)
                nvals = None
                if testing and persistent:
                    nvals = executable.transform(df[col])
                else:
                    nvals = executable.fit_transform(df[col])
                if isinstance(nvals, scipy.sparse.csr.csr_matrix):
                    nvals = nvals.todense()
                #fcols = [(col.format() + "_" + feature) for feature in executable.get_feature_names()]
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            elif self.media_type == VariableFileType.IMAGE:
                image_tensor = self._as_tensor(df[col].values)
                nvals = executable.transform(image_tensor)
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            if not (testing and persistent):
                primitive.executables[col] = executable
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
        # We look for the .csv.gz file by default. If it doesn't exist, try to load the .csv file
        if not os.path.exists(csvfile) and csvfile.endswith('.gz'):
            csvfile = csvfile[:-3]

        # Read the csv file while specifying the index column
        df = pd.read_csv(csvfile, index_col=indexcol)
        #df = df.reindex(pd.RangeIndex(df.index.max()+1)).ffill()
        df = df.reset_index(drop=True)

        colnames = []
        for col in cols:
            colnames.append(col['varName'])

        # Remove columns not specified
        for colname, col in df.iteritems():
            if colname not in colnames:
                df.drop(colname, axis=1, inplace=True)

        # Check for nested tabular data files, and load them in
        for col in cols:
            colname = col['varName']
            if col['varRole'] == 'file':
                for index, row in df.iterrows():
                    filename = row[colname]
                    if self.media_type is VariableFileType.TABULAR:
                        if not filename in self.nested_table:
                            nested_df = self.read_data(self.directory + '/raw_data/' + df.loc[index, colname] + ".gz", [], None)
                            self.nested_table[filename] = nested_df

        # Check all columns for special roles
        for col in cols:
            colname = col['varName']
            if col['varRole'] == 'file':
                # If the role is "file", then load in the raw data files
                for index, row in df.iterrows():
                    filepath = self.directory + '/raw_data/' + row[colname]
                    if self.media_type == VariableFileType.TEXT:
                        # Plain data load for text files
                        with open(filepath, 'rb') as myfile:
                            txt = myfile.read()
                            df.set_value(index, colname, txt)
                    elif self.media_type == VariableFileType.IMAGE:
                        # Load image files using keras with a standard target size
                        # TODO: Make the (224, 224) size configurable
                        from keras.preprocessing import image
                        df.set_value(index, colname, image.load_img(filepath, target_size=(224, 224)))
            if col['varRole'] == 'index' and colname.endswith('_index'):
                filename_colname = colname[:-6]
                for index in range(df.shape[0]):
                    filename = row[filename_colname]
                    nested_data = NestedData(filename_colname, colname, filename, df.loc[index, colname],
                                             self.nested_table[filename])
                    df.set_value(index, colname, nested_data)
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

    def _process_args(self, args, task_type, metric):
        result_args = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith('*'):
                result_args.append(self._get_arg_value(arg, task_type, metric))
            else:
                result_args.append(arg)
        return result_args

    def _process_kwargs(self, kwargs, task_type, metric):
        result_kwargs = {}
        for key, arg in kwargs.items():
            if isinstance(arg, str) and arg.startswith('*'):
                result_kwargs[key] = self._get_arg_value(arg, task_type, metric)
            else:
                result_kwargs[key] = arg
        return result_kwargs

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

            try:
                primfile = "%s/models/%s.%s.pkl" % (self.outputdir, pipeid, primid)
                statements.append("%s = sklearn.externals.joblib.load('%s')" % (primid, primfile))
                joblib.dump(primitive, primfile)
            except Exception as e:
                sys.stderr.write("ERROR pickling %s : %s\n" % (primitive.name, e))

            # Initialize primitive
            if not primitive.is_persistent:
                mod, cls = primitive.cls.rsplit('.', 1)
                imports.append(mod)
                statements.append("args = %s" % primitive.init_args)
                statements.append("kwargs = %s" % primitive.init_kwargs)
                statements.append("%s.executables = %s(*args, **kwargs)" % (primid, primitive.cls))

            if primitive.task == "Modeling":
                statements.append("print(%s.executables.predict(testdata))" % primid)
            else:
                if primitive.task == "PreProcessing":
                    if primitive.column_primitive:
                        statements.append("for col in testdata.columns:")
                        if primitive.is_persistent:
                            statements.append("    primex = %s.executables.get(col, None)" % primid)
                        else:
                            statements.append("    primex = %s.executables" % primid)
                        statements.append("    if primex is not None:")
                        statements.append("        testdata[col] = primex.fit_transform(testdata[col])")
                    else:
                        statements.append("testdata = %s.executables.fit_transform(testdata)" % primid)

                elif primitive.task == "FeatureExtraction":
                    persistent = "False"
                    if primitive.is_persistent:
                        persistent = "True"
                    statements.append("testdata = hp.featurise(testdata, %s, True, %s)" % (primid, persistent))

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


class NestedData(object):
    def __init__(self, filename_column, index_column, filename, index, nested_data):
        self.filename_column = filename_column
        self.index_column = index_column
        self.filename = filename
        self.index = index
        self.nested_data = nested_data
