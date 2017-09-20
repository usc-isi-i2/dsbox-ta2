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
from dsbox.schema.profile_schema import DataProfileType as dpt
from dsbox.executer.execution import Execution
from dsbox.executer import pickle_patch

import scipy.sparse.csr

import os
import stopit
import inspect
import importlib
import traceback

from builtins import range

# For DSBox Imputation arguments
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.schema.data_profile import DataProfile
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer

# sklearn metric functions
import sklearn.metrics


REMOTE = False

class ExecutionHelper(object):
    def __init__(self, data_directory, outputdir, csvfile=None, schema_file=None):
        self.e = Execution()
        self.directory = os.path.abspath(data_directory)
        self.outputdir = os.path.abspath(outputdir)
        if schema_file is None:
            schema_file = self.directory + os.sep + "dataSchema.json"
        self.schema = self.load_json(schema_file)
        self.columns = self.schema['trainData']['trainData']
        self.targets = self.schema['trainData']['trainTargets']
        self.indexcol = self.get_index_column(self.columns)
        self.boundarycols = self.get_boundary_columns(self.columns)
        self.media_type = self.get_media_type(self.schema, self.columns)
        self.data = None
        self.nested_table = dict()
        if csvfile:
            self.data = self.read_data(self.directory + os.sep + csvfile, self.columns, self.indexcol)
        # To be set manually later (not via constructor)
        self.task_type = None
        self.task_subtype = None
        self.metric = None
        self.metric_function = None
        self.problemid = None
        self.tmp_dir = outputdir + os.sep + ".." + os.sep + "temp" # Default


    def instantiate_primitive(self, primitive):
        executable = None
        # Parse arguments
        args = []
        kwargs = {}
        if primitive.getInitKeywordArgs():
            kwargs = self._process_kwargs(
                primitive.getInitKeywordArgs(), self.task_type, self.metric_function)
        if primitive.getInitArgs():
            args = self._process_args(
                primitive.getInitArgs(), self.task_type, self.metric_function)

        # Instantiate primitive
        if REMOTE:
            executable = self.e.execute(primitive.cls, args=args, kwargs=kwargs)
        else:
            mod, cls = primitive.cls.rsplit('.', 1)
            try:
                module = importlib.import_module(mod)
                PrimitiveClass = getattr(module, cls)
                executable = PrimitiveClass(*args, **kwargs)
            except Exception as e:
                sys.stderr.write("ERROR _instantiate_primitive(%s) : %s\n" % (primitive, e))
                traceback.print_exc()
                return None

        # Check if the executable is a unified interface executable
        primitive.unified_interface = True
        try:
            executable.set_training_data()
        except AttributeError:
            # No method set_training_data. This is not a unified interface primitive
            primitive.unified_interface = False
        except:
            pass
        return executable


    @stopit.threading_timeoutable()
    def execute_primitive(self, primitive, df, df_lbl, cur_profile=None):
        persistent = primitive.is_persistent
        indices = df.index
        if primitive.column_primitive:
            # print(df.columns)
            # A primitive that is run per column
            for col in df.columns:
                colname = col.format()
                colprofile = None
                if cur_profile is not None:
                    colprofile = cur_profile.columns[colname]

                if self._profile_matches_precondition(primitive.preconditions, colprofile) and not colprofile[dpt.LIST]:
                    try:
                        executable = self.instantiate_primitive(primitive)

                        # FIXME: Hack for Label encoder for python3 (cannot handle missing values)
                        if (primitive.name == "Label Encoder") and (sys.version_info[0] == 3):
                            df[col] = df[col].fillna('')
                        (df[col], executable) = self._execute_primitive(
                            primitive, executable, df[col], None, False, persistent)
                        primitive.executables[colname] = executable
                    except Exception as e:
                        sys.stderr.write("ERROR execute_primitive(%s): %s\n" % (primitive, e))
                        traceback.print_exc()
                        return None
        else:
            primitive.executables = self.instantiate_primitive(primitive)
            if self._profile_matches_precondition(primitive.preconditions, cur_profile.profile):
                (df, executable) = self._execute_primitive(
                    primitive, primitive.executables, df, df_lbl, False, persistent)
                primitive.executables = executable

        return pd.DataFrame(df, index=indices)


    @stopit.threading_timeoutable()
    def test_execute_primitive(self, primitive, df):
        persistent = primitive.is_persistent
        indices = df.index
        if primitive.column_primitive:
            # A primitive that is run per column
            for col in df.columns:
                colname = col.format()
                # If during test phase, this column washn't hash
                if colname not in primitive.executables.keys():
                    continue

                executable = None
                if not persistent:
                    executable = self.instantiate_primitive(primitive)
                else:
                    executable = primitive.executables.get(colname, None)

                try:
                    # FIXME: Hack for Label encoder for python3 (cannot handle missing values)
                    if (primitive.name == "Label Encoder") and (sys.version_info[0] == 3):
                        df[col] = df[col].fillna('')
                    #print("- Test on column: %s" % colname)
                    (df[col], executable) = self._execute_primitive(
                        primitive, executable, df[col], None, True, persistent)
                except Exception as e:
                    sys.stderr.write("ERROR execute_primitive(%s): %s\n" % (primitive, e))
                    traceback.print_exc()
                    return None
        else:
            if not persistent:
                primitive.executables = self.instantiate_primitive(primitive)
            (df, executable) = self._execute_primitive(
                primitive, primitive.executables, df, None, True, persistent)

        return pd.DataFrame(df, index=indices)

    def _profile_matches_precondition(self, preconditions, profile):
        for precondition in preconditions.keys():
            prec_value = preconditions[precondition]
            if prec_value != profile.get(precondition, None):
                return False
        return True

    def _execute_primitive(self, primitive, executable, df, df_lbl, testing, persistent):
        args = [df]
        if df_lbl is not None:
            args.append(df_lbl)
        retval = None
        if primitive.unified_interface:
            if (testing and persistent):
                retval = executable.produce(inputs=df)
            else:
                executable.set_training_data(inputs=df, outputs=df_lbl)
                executable.fit()
                retval = executable.produce(inputs=df)
        else:
            if (testing and persistent):
                if REMOTE:
                    retval = self.e.execute('transform', args=args, kwargs=None, obj=executable)
                else:
                    retval = executable.transform(*args)
            else:
                if REMOTE:
                    retval = self.e.execute('fit_transform', args=args, kwargs=None, obj=executable)
                else:
                    retval = executable.fit_transform(*args)

            if persistent and not testing:
                if REMOTE:
                    executable = self.e.execute('fit', args=args, kwargs=None, obj=executable, objreturn=True)
                else:
                    executable = executable.fit(*args)
        return (retval, executable)

    @stopit.threading_timeoutable()
    def cross_validation_score(self, primitive, X, y, cv=4):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        vals = []

        for k, (train, test) in enumerate(kf.split(X, y)):
            executable = self.instantiate_primitive(primitive)
            trainX = X.take(train, axis=0)
            trainY = y.take(train, axis=0).values.ravel()
            testX = X.take(test, axis=0)
            testY = y.take(test, axis=0).values.ravel()

            if primitive.unified_interface:
                executable.set_training_data(inputs=trainX, outputs=trainY)
                executable.fit()
                ypred = executable.produce(inputs=testX)
            else:
                if REMOTE:
                    prim = self.e.execute('fit', args=[trainX, trainY], kwargs=None, obj=executable, objreturn=True)
                    ypred = self.e.execute('predict', args=[testX], kwargs=None, obj=executable)
                else:
                    executable.fit(trainX, trainY)
                    ypred = executable.predict(testX)

            #print ("Trained on {} samples, Tested on {} samples".format(len(train), len(ypred)))
            val = self._call_function(self.metric_function, testY, ypred)
            vals.append(val)

        # fit the model finally over the whole training data for evaluation later over actual test data
        executable = self.instantiate_primitive(primitive)
        if primitive.unified_interface:
            executable.set_training_data(inputs=X, outputs=y.values.ravel())
            executable.fit()
        else:
            if REMOTE:
                executable = self.e.execute('fit', args=[X, y], kwargs=None, obj=executable, objreturn=True)
            else:
                executable.fit(X, y.values.ravel())
        primitive.executables = executable

        #print ("Returning {} : {}".format(vals, np.average(vals)))
        return np.average(vals)

    def _as_tensor(self, image_list):
        from keras.preprocessing import image
        shape = (len(image_list), ) + image.img_to_array(image_list[0]).shape
        result = np.empty(shape)
        for i in range(len(image_list)):
            result[i] = image.img_to_array(image_list[i])
        return result

    @stopit.threading_timeoutable()
    def featurise(self, primitive, df):
        persistent = primitive.is_persistent
        ncols = [col.format() for col in df.columns]
        featurecols = self.raw_data_columns(self.columns)
        indices = df.index
        for col in featurecols:
            if self.media_type == VariableFileType.TEXT:
                executable = self.instantiate_primitive(primitive)

                # Using an unfitted primitive for each column (needed for Corex)
                #df_col = pd.DataFrame(df[col])
                #executable.fit(df_col)
                nvals = executable.fit_transform(df[col])
                if persistent:
                    executable.fit(df[col])
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
                executable = self.instantiate_primitive(primitive)

                image_tensor = self._as_tensor(df[col].values)
                nvals = executable.transform(image_tensor)
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            elif self.media_type == VariableFileType.AUDIO:
                # Featurize audio
                fcols = []
                for idx, row in df.iterrows():
                    if row[col] is None:
                        continue
                    (audio_clip, sampling_rate) = row[col]
                    start = None
                    end = None
                    bcols = self.boundarycols
                    if len(bcols) == 2:
                        start = int(sampling_rate * float(row[bcols[0]]))
                        end = int(sampling_rate * float(row[bcols[1]]))
                        if start > end:
                            tmp = start
                            start = end
                            end = tmp
                        audio_clip = audio_clip[start:end]

                    primitive.init_kwargs['sampling_rate'] = sampling_rate
                    executable = self.instantiate_primitive(primitive)
                    executable.fit('time_series', [audio_clip])
                    nvals = executable.transform('array2+N')
                    features = nvals[1]

                    allfeatures = {}
                    for feature in features:
                        for index in range(0, len(feature)):
                            fcol = col.format() + "_" + str(index)
                            featurevals = allfeatures.get(fcol, [])
                            featurevals.append(feature[index])
                            allfeatures[fcol] = featurevals

                    for fcol in allfeatures.keys():
                        if df.get(fcol) is None:
                            fcols.append(fcol)
                            df.set_value(idx, fcol, 0)
                            #df[fcol] = df[fcol].astype(object)
                        df.set_value(idx, fcol, np.average(allfeatures[fcol]))

                del df[col]
                if len(self.boundarycols) == 2:
                    del df[self.boundarycols[0]]
                    del df[self.boundarycols[1]]

            primitive.executables[col] = executable

        return pd.DataFrame(df, index=indices)

    @stopit.threading_timeoutable()
    def test_featurise(self, primitive, df):
        persistent = primitive.is_persistent
        ncols = [col.format() for col in df.columns]
        featurecols = self.raw_data_columns(self.columns)
        indices = df.index
        for col in featurecols:
            executable = None
            if not persistent:
                executable = self.instantiate_primitive(primitive)
            else:
                executable = primitive.executables[col]

            if self.media_type == VariableFileType.TEXT:
                # Using an unfitted primitive for each column (needed for Corex)
                #df_col = pd.DataFrame(df[col])
                #executable.fit(df_col)
                nvals = None
                if persistent:
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
            elif self.media_type == VariableFileType.AUDIO:
                # Featurize audio
                fcols = []
                for idx, row in df.iterrows():
                    if row[col] is None:
                        continue
                    (audio_clip, sampling_rate) = row[col]
                    start = None
                    end = None
                    bcols = self.boundarycols
                    if len(bcols) == 2:
                        start = int(sampling_rate * float(row[bcols[0]]))
                        end = int(sampling_rate * float(row[bcols[1]]))
                        if start > end:
                            tmp = start
                            start = end
                            end = tmp
                        audio_clip = audio_clip[start:end]

                    primitive.init_kwargs['sampling_rate'] = sampling_rate
                    executable = self.instantiate_primitive(primitive)
                    executable.fit('time_series', [audio_clip])
                    nvals = executable.transform('array2+N')
                    features = nvals[1]

                    allfeatures = {}
                    for feature in features:
                        for i in range(0, len(feature)):
                            fcol = col.format() + "_" + str(i)
                            featurevals = allfeatures.get(fcol, [])
                            featurevals.append(feature[i])
                            allfeatures[fcol] = featurevals

                    for fcol in allfeatures.keys():
                        if df.get(fcol) is None:
                            fcols.append(fcol)
                            df.set_value(idx, fcol, 0)
                            #df[fcol] = df[fcol].astype(object)
                        df.set_value(idx, fcol, np.average(allfeatures[fcol]))

                del df[col]
                if len(self.boundarycols) == 2:
                    del df[self.boundarycols[0]]
                    del df[self.boundarycols[1]]

        return pd.DataFrame(df, index=indices)

    """
    Set the task type and task subtype
    """
    def set_task_type(self, task_type, task_subtype=None):
        self.task_type = TaskType(task_type)
        self.task_subtype = None
        if task_subtype is not None:
            task_subtype = task_subtype.replace(self.task_type.value.title(), "")
            self.task_subtype = TaskSubType(task_subtype)

    """
    Set the metric
    """
    def set_metric(self, metric):
        metric = metric[0].lower() + metric[1:]
        self.metric = Metric(metric)
        self.metric_function = self._get_metric_function(self.metric)

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

    def get_boundary_columns(self, columns):
        cols = []
        for col in columns:
            if col.get('varRole', None) == 'boundary':
                cols.append(col['varName'])
        return cols

    def load_json(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d

    def read_data(self, csvfile, cols, indexcol, labeldata=False):
        # We look for the .csv.gz file by default. If it doesn't exist, try to load the .csv file
        if not os.path.exists(csvfile) and csvfile.endswith('.gz'):
            csvfile = csvfile[:-3]

        # Read the csv file
        df = pd.read_csv(csvfile)

        # Filter columns if specified
        if len(cols) > 0:
            colnames = []
            for col in cols:
                colnames.append(col['varName'])

            # Remove columns not specified
            for colname, col in df.iteritems():
                if colname not in colnames:
                    df.drop(colname, axis=1, inplace=True)

            # Check for nested tabular data files, and load them in
            tabular_columns = []
            index_columns = []
            for col in cols:
                colname = col['varName']
                varRole = col.get('varRole', None)
                varType = col.get('varType', None)
                varFileType = col.get('varFileType', None)
                if varRole == 'index':
                    index_columns.append(colname)
                if varType == 'file' and varFileType == 'tabular':
                    tabular_columns.append(colname)
                    filename = df.loc[:, colname].unique()
                    if len(filename) > 1:
                        raise AssertionError('Expecting one unique filename per column: {}'.format(colname))
                    filename = filename[0]
                    if not filename in self.nested_table:
                        csvfile = self.directory + os.sep + 'raw_data' + os.sep + filename
                        if not os.path.exists(csvfile):
                            csvfile += '.gz'
                        nested_df = self.read_data(csvfile, [], None)
                        self.nested_table[filename] = nested_df


            # Match index columns to tabular columns
            if len(tabular_columns) == len(index_columns) - 1:
                # New r_32 dataset has two tabular columns and three index columns. Need to remove d3mIndex
                # New r_26 dataset has exactly one tabular column and one index column (d3mIndex)
                index_columns = index_columns[1:]
            if not len(tabular_columns) == len(index_columns):
                raise AssertionError('Number tabular and index columns do not match: {} != {}'
                                     .format(len(tabular_columns), (index_columns)))

            # Check all columns for special roles
            for col in cols:
                colname = col['varName']
                varRole = col.get('varRole', None)
                varType = col.get('varType', None)
                varFileType = col.get('varFileType', None)
                if varRole == 'file' or varType == 'file':
                    # If the role is "file", then load in the raw data files
                    if self.media_type in (VariableFileType.TEXT, VariableFileType.IMAGE, VariableFileType.AUDIO):
                        for index, row in df.iterrows():
                            filepath = self.directory + os.sep + 'raw_data' + os.sep + row[colname]
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
                            elif self.media_type == VariableFileType.AUDIO:
                                # Load audio files
                                import librosa
                                # Load file
                                try:
                                    print (filepath)
                                    audiodata = librosa.load(filepath, sr=None)
                                    df.set_value(index, colname, audiodata)
                                except Exception as e:
                                    df.set_value(index, colname, None)

            for file_colname, index_colname in zip(tabular_columns, index_columns):
                # FIXME: Assumption here that all entries for the filename are the same per column
                filename = df.iloc[0][file_colname]

                # Merge the nested table with parent table on the index column
                nested_table = self.nested_table[filename]
                df = pd.merge(df, nested_table, on=index_colname)

                # Remove file and index columns since the content has been replaced
                del df[file_colname]
                if index_colname != self.indexcol:
                    del df[index_colname]
                ncols = []
                for col in cols:
                    if col['varName'] not in [file_colname, index_colname]:
                        ncols.append(col)
                cols = ncols
                # Add nested table columns
                for nested_colname in nested_table.columns:
                    if not nested_colname == index_colname:
                        cols.append({'varName': nested_colname})

            if cols:
                if labeldata:
                    self.targets = cols
                else:
                    self.columns = cols

        if indexcol is not None:
            # Set the table's index column
            df = df.set_index(indexcol, drop=True)

            if not labeldata:
                # Check if we need to set the media type for any columns
                profile = DataProfile(df)
                if profile.profile[dpt.TEXT]:
                    if self.media_type == VariableFileType.TABULAR or self.media_type is None:
                        self.media_type = VariableFileType.TEXT
                        for colname in profile.columns.keys():
                            colprofile = profile.columns[colname]
                            if colprofile[dpt.TEXT]:
                                for col in self.columns:
                                    if col['varName'] == colname:
                                        col['varType'] = 'file'
                                        col['varFileType'] = 'text'
                                        col['varFileFormat'] = 'text/plain'

        return df

    def get_media_type(self, schema, cols):
        if schema.get('rawData', False):
            types = schema.get('rawDataFileTypes', {})
            for ext in types.keys():
                typ = types[ext]
                return self._mime_to_media_type(typ)

            # Check for nested tabular data files, and load them in
            for col in cols:
                varType = col.get('varType', None)
                if varType == 'file':
                    return VariableFileType(col.get('varFileType'))
        return None

    def _mime_to_media_type(self, mime):
        if mime.startswith("text/csv"):
            return VariableFileType.TABULAR
        elif mime.startswith("text"):
            return VariableFileType.TEXT
        elif mime.startswith("image"):
            return VariableFileType.IMAGE
        elif mime.startswith("audio"):
            return VariableFileType.AUDIO
        elif mime.startswith("video"):
            return VariableFileType.VIDEO
        return None

    def raw_data_columns(self, columns):
        cols = []
        for col in columns:
            varRole = col.get('varRole', None)
            varType = col.get('varType', None)
            if varRole == 'file' or varType == 'file':
                cols.append(col['varName'])
        return cols

    def _process_args(self, args, task_type, metric):
        result_args = []
        for arg in args:
            if (isinstance(arg, str) or isinstance(arg, unicode)) and arg.startswith('*'):
                result_args.append(self._get_arg_value(arg, task_type, metric))
            else:
                result_args.append(arg)
        return result_args

    def _get_arg_value(self, arg_specification, task_type, metric):
        if not arg_specification.startswith('*'):
            return arg_specification
        arg_specification = arg_specification[1:]
        if arg_specification == "SCORER":
            return make_scorer(metric, greater_is_better=True)
        elif arg_specification == "LOSS":
            return make_scorer(metric, greater_is_better=False)
        elif arg_specification == "ESTIMATOR":
            if task_type == TaskType.CLASSIFICATION:
                return LogisticRegression()
            elif task_type == TaskType.REGRESSION:
                return LinearRegression()
            else:
                raise Exception("Not yet implemented: Arg specification ESTIMATOR task type: {}"
                                .format(task_type))
        else:
            raise Exception(
                "Unkown Arg specification: {}".format(arg_specification))

    def _process_kwargs(self, kwargs, task_type, metric):
        result_kwargs = {}
        for key, arg in kwargs.items():
            if isinstance(arg, str) and arg.startswith('*'):
                result_kwargs[key] = self._get_arg_value(arg, task_type, metric)
            else:
                result_kwargs[key] = arg
        return result_kwargs

    def create_pipeline_executable(self, pipeline):
        pipeid = pipeline.id

        modelsdir = self.outputdir + os.sep + "models"
        if not os.path.exists(modelsdir):
            os.makedirs(modelsdir)

        rdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        imports = []
        statements = [
                "import sys",
                "sys.path.append('%s')" % rdir,
                "from dsbox_dev_setup import path_setup",
                "path_setup()",
                "",
                "import json",
                "import numpy",
                "import pandas",
                "import os.path",
                ""
                "import sklearn.externals",
                "from dsbox.executer.executionhelper import ExecutionHelper",
                "from dsbox.schema.problem_schema import TaskType, Metric",
                "",
                "# Pipeline : %s" % str(pipeline),
                ""]

        statements.append("\ncurdir = os.path.dirname(os.path.abspath(__file__))")
        statements.append("numpy.set_printoptions(threshold=numpy.nan)")

        statements.append("\n# Defaults unless overridden by config json")
        resultsdir = os.path.abspath(self.outputdir + os.sep + ".." + os.sep + "results")
        statements.append("test_data_root = '%s'" % self.directory)
        statements.append("results_path = '%s%s%s.csv'" % (resultsdir, os.sep, pipeid))
        statements.append("executables_root = curdir")
        statements.append("temp_storage_root = '%s'" % self.tmp_dir)

        statements.append("\nconfig = {}")
        statements.append("if len(sys.argv) > 1:")
        statements.append("    with open(sys.argv[1]) as conf_data:")
        statements.append("        config = json.load(conf_data)")
        statements.append("        conf_data.close()")
        statements.append("if config.get('test_data_root', None) is not None:")
        statements.append("    test_data_root = config['test_data_root']")
        statements.append("if config.get('results_path', None) is not None:")
        statements.append("    results_path = config['results_path']")
        statements.append("if config.get('executables_root', None) is not None:")
        statements.append("    executables_root = config['executables_root']")
        statements.append("if config.get('temp_storage_root', None) is not None:")
        statements.append("    temp_storage_root = config['temp_storage_root']")

        statements.append("\nprint('Loading Data..')")
        statements.append("hp = ExecutionHelper(test_data_root, temp_storage_root, 'testData.csv.gz')" % self.directory)
        statements.append("hp.task_type = %s" % self.task_type)
        statements.append("hp.metric = %s" % self.metric)
        statements.append("hp.metric_function = hp._get_metric_function(hp.metric)")
        statements.append("testdata = hp.data")
        index = 1
        for primitive in pipeline.primitives:
            primid = "primitive_%s" % str(index)

            try:
                statements.append("\nprint('\\nExecuting %s...')" % primitive)
                # Remove executables(instances) from not persistent primitives
                # as many of them have pickling(serialization) issues
                execs = primitive.executables
                if not primitive.is_persistent:
                    if primitive.column_primitive:
                        execs = {}
                        for colname in primitive.executables.keys():
                            execs[colname] = primitive.executables[colname]
                            primitive.executables[colname] = None
                    else:
                        execs = primitive.executables
                        primitive.executables = None

                primfilename = "models%s%s.%s.pkl" % (os.sep, pipeid, primid)
                primfile = "%s%s%s" % (self.outputdir, os.sep, primfilename)
                statements.append("primfile = executables_root + '%s%s'" % (os.sep, primfilename))
                statements.append("%s = sklearn.externals.joblib.load(primfile)" % primid)
                joblib.dump(primitive, primfile)

                # Restore executables(instances) after pickling(serialization) is done
                primitive.executables = execs
            except Exception as e:
                sys.stderr.write("ERROR pickling %s : %s\n" % (primitive.name, e))

            if primitive.task == "Modeling":
                # Initialize primitive
                if not primitive.is_persistent:
                    mod, cls = primitive.cls.rsplit('.', 1)
                    imports.append(mod)
                    statements.append("args = %s" % primitive.init_args)
                    statements.append("kwargs = %s" % primitive.init_kwargs)
                    statements.append("%s.executables = %s(*args, **kwargs)" % (primid, primitive.cls))

                statements.append("\nprint('\\nStoring results in %s' % results_path)")
                statements.append("resultsdir = os.path.dirname(results_path)")
                statements.append("if not os.path.exists(resultsdir):")
                statements.append("    os.makedirs(resultsdir)")
                if primitive.unified_interface:
                    statements.append("result = pandas.DataFrame(%s.executables.produce(inputs=testdata), index=testdata.index, columns=['%s'])" %
                        (primid, self.targets[1]['varName']))
                else:
                    statements.append("result = pandas.DataFrame(%s.executables.predict(testdata), index=testdata.index, columns=['%s'])" %
                        (primid, self.targets[1]['varName']))
                statements.append("result.to_csv(results_path, index_label='%s')" % self.targets[0]['varName'])
            else:
                if primitive.task == "PreProcessing":
                    statements.append("testdata = hp.test_execute_primitive(%s, testdata)" % primid)
                elif primitive.task == "FeatureExtraction":
                    statements.append("testdata = hp.test_featurise(%s, testdata)" % primid)

            index += 1

        # Write executable
        exfilename = "%s%s%s" % (self.outputdir, os.sep, pipeid)
        with open(exfilename, 'a') as exfile:
            exfile.write("#!/usr/bin/env python3\n\n")
            for imp in set(imports):
                exfile.write("import %s\n" % imp)
            for st in statements:
                exfile.write(st+"\n")
        os.chmod(exfilename, 0o755)


    def _get_metric_function(self, metric):
        if metric==Metric.ACCURACY:
            return sklearn.metrics.accuracy_score
        elif metric==Metric.F1:
            return sklearn.metrics.f1_score
        elif metric==Metric.F1_MICRO:
            return self.f1_micro
        elif metric==Metric.F1_MACRO:
            return self.f1_macro
        elif metric==Metric.ROC_AUC:
            return sklearn.metrics.roc_auc_score
        elif metric==Metric.ROC_AUC_MICRO:
            return self.roc_auc_micro
        elif metric==Metric.ROC_AUC_MACRO:
            return self.roc_auc_macro
        elif metric==Metric.MEAN_SQUARED_ERROR:
            return sklearn.metrics.mean_squared_error
        elif metric==Metric.ROOT_MEAN_SQUARED_ERROR:
            return self.root_mean_squared_error
        elif metric==Metric.ROOT_MEAN_SQUARED_ERROR_AVG:
            return self.root_mean_squared_error
        elif metric==Metric.MEAN_ABSOLUTE_ERROR:
            return sklearn.metrics.mean_absolute_error
        elif metric==Metric.R_SQUARED:
            return sklearn.metrics.r2_score
        elif metric==Metric.NORMALIZED_MUTUAL_INFORMATION:
            return sklearn.metrics.normalized_mutual_info_score
        elif metric==Metric.JACCARD_SIMILARITY_SCORE:
            return sklearn.metrics.jaccard_similarity_score
        return sklearn.metrics.accuracy_score

    ''' Custom Metric Functions '''
    def f1_micro(self, y_true, y_pred):
        return sklearn.metrics.f1_score(y_true, y_pred, average="micro")

    def f1_macro(self, y_true, y_pred):
        return sklearn.metrics.f1_score(y_true, y_pred, average="macro")

    def roc_auc_micro(self, y_true, y_pred):
        return sklearn.metrics.roc_auc_score(y_true, y_pred, average="micro")

    def roc_auc_macro(self, y_true, y_pred):
        return sklearn.metrics.roc_auc_score(y_true, y_pred, average="macro")

    def root_mean_squared_error(self, y_true, y_pred):
        import math
        return math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
