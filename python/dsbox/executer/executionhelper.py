import os
import sys
import json
import copy
import time
import uuid
import shutil
import os.path
import numpy as np
import pandas as pd


from primitive_interfaces.base import PrimitiveBase
from primitive_interfaces.generator import GeneratorPrimitiveBase
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from sklearn.externals import joblib
from sklearn.model_selection import KFold

from dsbox.schema.dataset_schema import VariableFileType
from dsbox.schema.profile_schema import DataProfileType as dpt
from dsbox.schema.problem_schema import TaskType
from dsbox.executer.execution import Execution
from dsbox.executer import pickle_patch

import scipy.sparse.csr

import stopit
import inspect
import importlib
import traceback

from builtins import range

# For DSBox Imputation arguments
from dsbox.schema.data_profile import DataProfile
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer

REMOTE = False

class ExecutionHelper(object):
    def __init__(self, data_manager, schema_manager):
        self.e = Execution()
        self.dm = data_manager
        self.sm = schema_manager

    def instantiate_primitive(self, primitive):
        executable = None
        # Parse arguments
        args = []
        kwargs = {}
        if primitive.getInitKeywordArgs():
            kwargs = self._process_kwargs(
                primitive.getInitKeywordArgs(), self.sm.task_type, self.sm.metric_functions)
        if primitive.getInitArgs():
            args = self._process_args(
                primitive.getInitArgs(), self.sm.task_type, self.sm.metric_functions)

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
        primitive.unified_interface = isinstance(executable, PrimitiveBase)

        return executable


    @stopit.threading_timeoutable()
    def execute_primitive(self, primitive, df, df_lbl, cur_profile=None):
        primitive.start_time = time.time()
        persistent = primitive.is_persistent
        indices = df.index
        try:
            if primitive.column_primitive:
                # print(df.columns)
                # A primitive that is run per column
                for col in df.columns:
                    colname = col.format()
                    colprofile = None
                    if cur_profile is not None:
                        colprofile = cur_profile.columns[colname]

                    if self._profile_matches_precondition(primitive.preconditions, colprofile) and not colprofile[dpt.LIST]:
                        executable = self.instantiate_primitive(primitive)
                        if executable is None:
                            primitive.finished = True
                            return None
                        # FIXME: Hack for Label encoder for python3 (cannot handle missing values)
                        if (primitive.name == "Label Encoder") and (sys.version_info[0] == 3):
                            df[col] = df[col].fillna('')
                        (df[col], executable) = self._execute_primitive(
                            primitive, executable, df[col], None, False, persistent)
                        primitive.executables[colname] = executable
            else:
                primitive.executables = self.instantiate_primitive(primitive)
                if primitive.executables is None:
                    primitive.finished = True
                    return None
                if self._profile_matches_precondition(primitive.preconditions, cur_profile.profile):
                    (df, executable) = self._execute_primitive(
                        primitive, primitive.executables, df, df_lbl, False, persistent)
                    primitive.executables = executable

        except Exception as e:
            try:
                sys.stderr.write("ERROR execute_primitive(%s): %s\n" % (primitive, e))
                traceback.print_exc()
                primitive.finished = True
            except:
                pass
            return None

        primitive.end_time = time.time()
        primitive.finished = True
        primitive.progress = 1.0
        primitive.pipeline.notifyChanges()

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

                if executable is None:
                    return None

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
            if primitive.executables is None:
                return None
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
                if isinstance(executable, SupervisedLearnerPrimitiveBase):
                    executable.set_training_data(inputs=df, outputs=df_lbl)
                elif isinstance(executable, UnsupervisedLearnerPrimitiveBase):
                    executable.set_training_data(inputs=df)
                elif isinstance(executable, GeneratorPrimitiveBase):
                    executable.set_training_data(outputs=df_lbl)
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
        primitive.start_time = time.time()

        kf = KFold(n_splits=cv, shuffle=True, random_state=int(time.time()))
        metric_values = {}

        tcols = [self.dm.data.target_columns[0]['varName']]
        yPredictions = None
        num = 0.0
        for k, (train, test) in enumerate(kf.split(X, y)):
            executable = self.instantiate_primitive(primitive)
            if executable is None:
                primitive.finished = True
                return None

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

            ypredDF = pd.DataFrame(ypred, index=testX.index, columns=tcols)
            if yPredictions is None:
                yPredictions = ypredDF
            else:
                yPredictions = pd.concat([yPredictions, ypredDF])

            num = num + 1.0
            # TODO: Removing this for now
            primitive.progress = num/cv
            primitive.pipeline.notifyChanges()

        yPredictions = yPredictions.sort_index()

        #print ("Trained on {} samples, Tested on {} samples".format(len(train), len(ypred)))
        for i in range(0, len(self.sm.metrics)):
            metric = self.sm.metrics[i]
            fn = self.sm.metric_functions[i]
            metric_val = self._call_function(fn, y, yPredictions)
            if metric_val is None:
                return None
            metric_values[metric.name] = metric_val

        # fit the model finally over the whole training data for evaluation later over actual test data
        executable = self.instantiate_primitive(primitive)
        if executable is None:
            primitive.finished = True
            return None

        if primitive.unified_interface:
            executable.set_training_data(inputs=X, outputs=y.values.ravel())
            executable.fit()
        else:
            if REMOTE:
                executable = self.e.execute('fit', args=[X, y], kwargs=None, obj=executable, objreturn=True)
            else:
                executable.fit(X, y.values.ravel())
        primitive.executables = executable

        primitive.end_time = time.time()
        primitive.progress = 1.0
        primitive.finished = True
        primitive.pipeline.notifyChanges()

        #print ("Returning {}".format(metric_values))
        return (yPredictions, metric_values)

    def _as_tensor(self, image_list):
        from keras.preprocessing import image
        shape = (len(image_list), ) + image.img_to_array(image_list[0]).shape
        result = np.empty(shape)
        for i in range(len(image_list)):
            result[i] = image.img_to_array(image_list[i])
        return result

    @stopit.threading_timeoutable()
    def featurise(self, primitive, df):
        primitive.start_time = time.time()

        persistent = primitive.is_persistent
        ncols = [col.format() for col in df.columns]
        featurecols = self.raw_data_columns(self.dm.data.input_columns)
        indices = df.index
        for col in featurecols:
            if self.dm.data.media_type == VariableFileType.TEXT:
                executable = self.instantiate_primitive(primitive)
                if executable is None:
                    primitive.finished = True
                    return None

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
            elif self.dm.data.media_type == VariableFileType.IMAGE:
                executable = self.instantiate_primitive(primitive)
                if executable is None:
                    primitive.finished = True
                    return None
                image_tensor = self._as_tensor(df[col].values)
                nvals = executable.transform(image_tensor)
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            elif self.dm.data.media_type == VariableFileType.AUDIO:
                # Featurize audio
                fcols = []
                for idx, row in df.iterrows():
                    if row[col] is None:
                        continue
                    (audio_clip, sampling_rate) = row[col]
                    primitive.init_kwargs['sampling_rate'] = int(sampling_rate)
                    executable = self.instantiate_primitive(primitive)
                    if executable is None:
                        primitive.finished = True
                        return None
                    #executable.fit('time_series', [audio_clip])
                    features = executable.produce([audio_clip])[0]

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
                bcols = self.dm.data.boundary_columns
                if len(bcols) == 2:
                    del df[bcols[0]]
                    del df[bcols[1]]

            primitive.executables[col] = executable

        primitive.end_time = time.time()
        primitive.progress = 1.0
        primitive.finished = True
        primitive.pipeline.notifyChanges()

        return pd.DataFrame(df, index=indices)

    @stopit.threading_timeoutable()
    def test_featurise(self, primitive, df):
        persistent = primitive.is_persistent
        ncols = [col.format() for col in df.columns]
        featurecols = self.raw_data_columns(self.dm.data.input_columns)
        indices = df.index
        for col in featurecols:
            executable = None
            if not persistent:
                executable = self.instantiate_primitive(primitive)
            else:
                executable = primitive.executables[col]
            if executable is None:
                return None

            if self.dm.data.media_type == VariableFileType.TEXT:
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
            elif self.dm.data.media_type == VariableFileType.IMAGE:
                image_tensor = self._as_tensor(df[col].values)
                nvals = executable.transform(image_tensor)
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            elif self.dm.data.media_type == VariableFileType.AUDIO:
                # Featurize audio
                fcols = []
                for idx, row in df.iterrows():
                    if row[col] is None:
                        continue
                    (audio_clip, sampling_rate) = row[col]
                    primitive.init_kwargs['sampling_rate'] = sampling_rate
                    executable = self.instantiate_primitive(primitive)
                    if executable is None:
                        return None
                    features = executable.produce([audio_clip])[0]

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
                bcols = self.dm.data.boundary_columns
                if len(bcols) == 2:
                    del df[bcols[0]]
                    del df[bcols[1]]

        return pd.DataFrame(df, index=indices)

    def raw_data_columns(self, columns):
        cols = []
        for col in columns:
            varRole = col.get('varRole', None)
            varType = col.get('varType', None)
            if varRole == 'file' or varType == 'file':
                cols.append(col['varName'])
        return cols

    def _process_args(self, args, task_type, metrics):
        result_args = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith('*'):
                result_args.append(self._get_arg_value(arg, task_type, metrics))
            elif sys.version_info[0]==2 and isinstance(arg, unicode) and arg.startswith('*'):
                # For python 2 need to check for unicode strings
                result_args.append(self._get_arg_value(arg, task_type, metrics))
            else:
                result_args.append(arg)
        return result_args

    def _get_arg_value(self, arg_specification, task_type, metrics):
        if not arg_specification.startswith('*'):
            return arg_specification
        arg_specification = arg_specification[1:]
        if arg_specification == "SCORER":
            # Just use the first metric to make the scorer
            return make_scorer(metrics[0], greater_is_better=True)
        elif arg_specification == "LOSS":
            # Just use the first metric to make the scorer
            return make_scorer(metrics[0], greater_is_better=False)
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

    def _process_kwargs(self, kwargs, task_type, metrics):
        result_kwargs = {}
        for key, arg in kwargs.items():
            if isinstance(arg, str) and arg.startswith('*'):
                result_kwargs[key] = self._get_arg_value(arg, task_type, metrics)
            else:
                result_kwargs[key] = arg
        return result_kwargs

    def create_pipeline_executable(self, pipeline, config):
        pipeid = pipeline.id

        # Get directory information
        train_dir = config['training_data_root']
        exec_dir = config['executables_root']
        tmp_dir = config['temp_storage_root']

        # Copy over the data schema
        # FIXME: Deal with multiple schemas (when dealing with multiple data uris from TA3)
        orig_data_schema = config['dataset_schema']
        data_schema_file = str(uuid.uuid4()) + ".json"
        shutil.copyfile(orig_data_schema, tmp_dir + os.sep + data_schema_file)


        modelsdir = exec_dir + os.sep + "models"
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
                "from dsbox.planner.common.data_manager import DataManager",
                "from dsbox.planner.common.schema_manager import SchemaManager",
                "from dsbox.schema.problem_schema import TaskType, Metric",
                "",
                "# Pipeline : %s" % str(pipeline),
                ""]

        statements.append("\ncurdir = os.path.dirname(os.path.abspath(__file__))")
        statements.append("numpy.set_printoptions(threshold=numpy.nan)")

        statements.append("\n# Defaults unless overridden by config json")
        resultsdir = os.path.abspath(exec_dir + os.sep + ".." + os.sep + "results")
        statements.append("test_data_root = '%s'" % train_dir)
        statements.append("results_path = '%s%s%s.csv'" % (resultsdir, os.sep, pipeid))
        statements.append("executables_root = curdir")
        statements.append("temp_storage_root = '%s'" % tmp_dir)

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

        statements.append("\nsm = SchemaManager()")
        statements.append("sm.task_type = %s" % self.sm.task_type)
        metricstrs = "["
        for i in range(0, len(self.sm.metrics)):
            if i > 0:
                metricstrs += ","
            metricstrs += str(self.sm.metrics[i])
        metricstrs += "]"
        statements.append("sm.metrics = %s" % metricstrs)
        statements.append("sm.set_metric_functions()")

        statements.append("\ndm = DataManager()")
        statements.append("dm.initialize_test_data_from_defaults(temp_storage_root + '/%s', test_data_root)" % data_schema_file)
        statements.append("testdata = dm.data.input_data")

        statements.append("\nhp = ExecutionHelper(dm, sm)")
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
                primfile = "%s%s%s" % (exec_dir, os.sep, primfilename)
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
                target_column = self.dm.data.target_columns[0]['varName']
                if primitive.unified_interface:
                    statements.append("result = pandas.DataFrame(%s.executables.produce(inputs=testdata), index=testdata.index, columns=['%s'])" %
                        (primid, target_column))
                else:
                    statements.append("result = pandas.DataFrame(%s.executables.predict(testdata), index=testdata.index, columns=['%s'])" %
                        (primid, target_column))
                statements.append("result.to_csv(results_path, index_label='%s')" % self.dm.data.index_column)
            else:
                if primitive.task == "PreProcessing":
                    statements.append("testdata = hp.test_execute_primitive(%s, testdata)" % primid)
                elif primitive.task == "FeatureExtraction":
                    statements.append("testdata = hp.test_featurise(%s, testdata)" % primid)

            index += 1

        # Write executable
        exfilename = "%s%s%s" % (exec_dir, os.sep, pipeid)
        with open(exfilename, 'a') as exfile:
            exfile.write("#!/usr/bin/env python\n\n")
            for imp in set(imports):
                exfile.write("import %s\n" % imp)
            for st in statements:
                exfile.write(st+"\n")
        os.chmod(exfilename, 0o755)

    def _call_function(self, scoring_function, *args):
        mod = inspect.getmodule(scoring_function)
        try:
            module = importlib.import_module(mod.__name__)
            return scoring_function(*args)
        except Exception as e:
            sys.stderr.write("ERROR _call_function %s: %s\n" % (scoring_function, e))
            traceback.print_exc()
            return None
