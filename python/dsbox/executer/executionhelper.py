import os
import sys
import json
import copy
import time
import uuid
import shutil
import os.path
import tempfile
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

# Import D3M Primitives
import d3m.primitives

REMOTE = False

class ExecutionHelper(object):
    problem = None
    dataset = None

    def __init__(self, problem, data_manager):
        self.e = Execution()
        self.problem = problem
        self.data_manager = data_manager

    def instantiate_primitive(self, primitive):
        executable = None
        if REMOTE:
            # FIXME: What to do for remote????
            # executable = self.e.execute(primitive.cls, args=args, kwargs=kwargs)
            pass
        else:
            mod, cls = primitive.cls.rsplit('.', 1)
            try:
                import importlib
                module = importlib.import_module(mod)
                PrimitiveClass = getattr(module, cls)
                if issubclass(PrimitiveClass, PrimitiveBase):
                    primitive.unified_interface = True
                    hyperparams_class = PrimitiveClass.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    executable = PrimitiveClass(hyperparams=hyperparams_class.defaults())
                else:
                    args = []
                    kwargs = {}
                    if primitive.getInitKeywordArgs():
                        kwargs = self._process_kwargs(
                            primitive.getInitKeywordArgs(), self.problem.task_type, self.problem.metric_functions)
                    if primitive.getInitArgs():
                        args = self._process_args(
                            primitive.getInitArgs(), self.problem.task_type, self.problem.metric_functions)
                    executable = PrimitiveClass(*args, **kwargs)
                    primitive.unified_interface = False
            except Exception as e:
                sys.stderr.write("ERROR: instantiate_primitive {}: {}\n".format(primitive.name, e))
                #sys.stderr.write("ERROR _instantiate_primitive(%s)\n" % (primitive.name))
                #traceback.print_exc()
                return None
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
                            if df[col].dtype == object:
                                df[col] = df[col].fillna('')
                            else:
                                df[col] = df[col].fillna(0)
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
                sys.stderr.write("ERROR: execute_primitive {}: {}\n".format(primitive.name, e))
                #sys.stderr.write("ERROR execute_primitive(%s): %s\n" % (primitive, e))
                #traceback.print_exc()
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
                # If during test phase, this column wasn't hash
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
                    sys.stderr.write("ERROR: execute_primitive {}: {}\n".format(primitive.name, e))
                    #sys.stderr.write("ERROR execute_primitive(%s): %s\n" % (primitive, e))
                    #traceback.print_exc()
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
                retval = executable.produce(inputs=df).value
            else:
                if isinstance(executable, SupervisedLearnerPrimitiveBase):
                    executable.set_training_data(inputs=df, outputs=df_lbl)
                elif isinstance(executable, UnsupervisedLearnerPrimitiveBase):
                    executable.set_training_data(inputs=df)
                elif isinstance(executable, GeneratorPrimitiveBase):
                    executable.set_training_data(outputs=df_lbl)
                executable.fit()
                retval = executable.produce(inputs=df).value
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
        print("Executing %s" % primitive.name)
        sys.stdout.flush()

        # Redirect stderr to an error file
        errorfile = tempfile.TemporaryFile(prefix=primitive.name)
        sys.stderr = errorfile

        primitive.start_time = time.time()

        kf = KFold(n_splits=cv, shuffle=True, random_state=int(time.time()))
        metric_values = {}

        tcols = [self.data_manager.target_columns[0]['colName']]
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

            try:
                if primitive.unified_interface:
                    executable.set_training_data(inputs=trainX, outputs=trainY)
                    executable.fit()
                    ypred = executable.produce(inputs=testX).value
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
            except Exception as e:
                sys.stderr.write("ERROR: cross_validation {}: {}\n".format(primitive.name, e))
                #traceback.print_exc(e)

        if num == 0:
            return (None, None)

        yPredictions = yPredictions.sort_index()

        #print ("Trained on {} samples, Tested on {} samples".format(len(train), len(ypred)))
        for i in range(0, len(self.problem.metrics)):
            metric = self.problem.metrics[i]
            fn = self.problem.metric_functions[i]
            metric_val = self._call_function(fn, y, yPredictions)
            if metric_val is None:
                return None
            metric_values[metric.name] = metric_val

        primitive.end_time = time.time()
        primitive.progress = 1.0
        primitive.finished = True
        primitive.pipeline.notifyChanges()

        #print ("Returning {}".format(metric_values))
        return (yPredictions, metric_values)

    def create_primitive_model(self, primitive, X, y):
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
        featurecols = self.raw_data_columns(self.data_manager.input_columns)
        indices = df.index
        for col in featurecols:
            executable = None
            if self.data_manager.media_type == VariableFileType.TEXT:
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
            elif self.data_manager.media_type == VariableFileType.TIMESERIES:
                executable = self.instantiate_primitive(primitive)
                if executable is None:
                    primitive.finished = True
                    return None
                executable.set_training_data(inputs=df[col].values, outputs=[])
                executable.fit()
                call_result = executable.produce(inputs=df[col].values)
                fcols = [(col.format() + "_" + str(index)) for index in range(0, call_result.value.shape[1])]
                newdf = pd.DataFrame(call_result.value, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns = ncols
            elif self.data_manager.media_type == VariableFileType.IMAGE:
                executable = self.instantiate_primitive(primitive)
                if executable is None:
                    primitive.finished = True
                    return None
                image_tensor = self._as_tensor(df[col].values)
                call_result = executable.produce(inputs=image_tensor)
                nvals = call_result.value
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            elif self.data_manager.media_type == VariableFileType.AUDIO:
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
                    features = executable.produce([audio_clip]).value[0]

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
                '''
                bcols = self.data_manager.boundary_columns
                if len(bcols) == 2:
                    del df[bcols[0]]
                    del df[bcols[1]]
                '''

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
        featurecols = self.raw_data_columns(self.data_manager.input_columns)
        indices = df.index
        for col in featurecols:
            executable = None
            if not persistent:
                executable = self.instantiate_primitive(primitive)
            else:
                executable = primitive.executables[col]
            if executable is None:
                return None

            if self.data_manager.media_type == VariableFileType.TEXT:
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
            elif self.data_manager.media_type == VariableFileType.IMAGE:
                image_tensor = self._as_tensor(df[col].values)
                call_result = executable.produce(image_tensor)
                nvals = call_result.value
                fcols = [(col.format() + "_" + str(index)) for index in range(0, nvals.shape[1])]
                newdf = pd.DataFrame(nvals, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            elif self.data_manager.media_type == VariableFileType.AUDIO:
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
                    features = executable.produce([audio_clip]).value[0]

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
                '''
                bcols = self.data_manager.boundary_columns
                if len(bcols) == 2:
                    del df[bcols[0]]
                    del df[bcols[1]]
                '''

        return pd.DataFrame(df, index=indices)

    def raw_data_columns(self, columns):
        cols = []
        for col in columns:
            if ("refersTo" in col or
                    col['colType'] == "string"):
                cols.append(col['colName'])
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
        problem_root = config['problem_root']
        exec_dir = config['executables_root']
        tmp_dir = config['temp_storage_root']
        results_dir = os.path.abspath(exec_dir + os.sep + ".." + os.sep + "results")

        # FIXME: Deal with multiple dataset schemas
        dataset_schema = config['dataset_schema']
        problem_schema = config['problem_schema']

        modelsdir = tmp_dir + os.sep + "models"
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
                "from dsbox.planner.common.data_manager import Dataset, DataManager",
                "from dsbox.planner.common.problem_manager import Problem",
                "from dsbox.schema.problem_schema import TaskType, Metric",
                "",
                "# Pipeline : %s" % str(pipeline),
                ""]

        statements.append("\ncurdir = os.path.dirname(os.path.abspath(__file__))")
        statements.append("numpy.set_printoptions(threshold=numpy.nan)")

        statements.append("\n# Defaults unless overridden by config json")
        statements.append("dataset_schema = '%s'" % dataset_schema)
        statements.append("problem_schema = '%s'" % problem_schema)
        statements.append("problem_root = '%s'" % problem_root)
        statements.append("test_data_root = '%s'" % train_dir)
        statements.append("results_root = '%s'" % results_dir)
        statements.append("executables_root = curdir")
        statements.append("temp_storage_root = '%s'" % tmp_dir)
        statements.append("numcpus = %s" % config.get('cpus'))
        statements.append("timeout = %s*60" % config.get('timeout'))
        statements.append("ram = '%s'" % config.get('ram'))

        statements.append("\nconfig = {}")
        statements.append("if len(sys.argv) > 1:")
        statements.append("    with open(sys.argv[1]) as conf_data:")
        statements.append("        config = json.load(conf_data)")
        statements.append("        conf_data.close()")
        statements.append("if config.get('dataset_schema', None) is not None:")
        statements.append("    dataset_schema = config['dataset_schema']")
        statements.append("if config.get('problem_schema', None) is not None:")
        statements.append("    problem_schema = config['problem_schema']")
        statements.append("if config.get('problem_root', None) is not None:")
        statements.append("    problem_root = config['problem_root']")
        statements.append("if config.get('test_data_root', None) is not None:")
        statements.append("    test_data_root = config['test_data_root']")
        statements.append("if config.get('results_root', None) is not None:")
        statements.append("    results_root = config['results_root']")
        statements.append("if config.get('executables_root', None) is not None:")
        statements.append("    executables_root = config['executables_root']")
        statements.append("if config.get('temp_storage_root', None) is not None:")
        statements.append("    temp_storage_root = config['temp_storage_root']")
        statements.append("predictions_file = os.path.join(results_root, '%s')" % self.problem.predictions_file)
        statements.append("scores_file = os.path.join(results_root, '%s')" % self.problem.scores_file)

        statements.append("\nproblem = Problem()")
        statements.append("problem.load_problem(problem_root, problem_schema)")
        statements.append("\ndataset = Dataset()")
        statements.append("dataset.load_dataset(test_data_root, dataset_schema)")
        statements.append("\ndata_manager = DataManager()")
        statements.append("data_manager.initialize_data(problem, [dataset], view='TEST')")
        statements.append("\ntestdata = data_manager.input_data")

        statements.append("\nhp = ExecutionHelper(problem, data_manager)")
        index = 1

        ensembling = pipeline.ensemble is not None
        n_pipelines = len(pipeline.ensemble.pipelines) if ensembling else 1
        ens_pipeline = pipeline

        if ensembling:
            statements.append("results = []")

        for pipe_i in range(n_pipelines):
            if ensembling:
                pipeline = ens_pipeline.ensemble.pipelines[pipe_i]

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

                    # Remove pipeline from pickling
                    pipe = primitive.pipeline
                    primitive.pipeline = None

                    joblib.dump(primitive, primfile)
                    # Restore pipeline after pickling is done
                    primitive.pipeline = pipe
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

                    #statements.append("\nprint('\\nStoring results in %s' % predictions_file)")
                    statements.append("if not os.path.exists(results_root):")
                    statements.append("    os.makedirs(results_root)")
                    target_column = self.data_manager.target_columns[0]['colName']
                    
                    if ensembling:
                        if primitive.unified_interface:
                            statements.append("results.append(pandas.DataFrame(%s.executables.produce(inputs=testdata), index=testdata.index, columns=['%s']))" %
                                (primid, target_column))
                        else:
                            statements.append("results.append(pandas.DataFrame(%s.executables.predict(testdata), index=testdata.index, columns=['%s']))" %
                                (primid, target_column))
                    else:
                        statements.append("\nprint('\\nStoring results in %s' % predictions_file)")
                        if primitive.unified_interface:
                            statements.append("result = pandas.DataFrame(%s.executables.produce(inputs=testdata), index=testdata.index, columns=['%s'])" %
                                (primid, target_column))
                        else:
                            statements.append("result = pandas.DataFrame(%s.executables.predict(testdata), index=testdata.index, columns=['%s'])" %
                                (primid, target_column))
                        statements.append("result.to_csv(predictions_file, index_label='%s')" % self.data_manager.index_column)

                else:
                    if primitive.task == "PreProcessing":
                        statements.append("testdata = hp.test_execute_primitive(%s, testdata)" % primid)
                    elif primitive.task == "FeatureExtraction":
                        statements.append("testdata = hp.test_featurise(%s, testdata)" % primid)

                index += 1
        
        # TO DO : update results_np after each pipeline run with weights up to weight i 
        #        (in order to have intermediate results in case test time runs out)
        if ensembling:
            statements.append("results_np = numpy.array([df.values for df in results])")
            # hacky way of passing weights array into python program
            weights_string = ', '.join([str(w) for w in ens_pipeline.ensemble.pipeline_weights])
            statements.append("weights_np = numpy.array([%s]).astype(numpy.float32)" % weights_string)
            # weighted average of predictions
            statements.append("average_pred = numpy.multiply(results_np, weights_np)/numpy.sum(weights_np)")
            # round if discrete metric (e.g. classification)
            if ens_pipeline.ensemble.discrete_metric:
                statements.append("average_pred = numpy.rint(average_pred)")
            # write
            statements.append("result = pandas.DataFrame(average_pred, index=testdata.index, columns=['%s'])" % self.data_manager.target_columns[0]['colName'])
            statements.append("result.to_csv(predictions_file, index_label='%s')" % self.data_manager.index_column)

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
            sys.stderr.write("ERROR: _call_function {}: {}\n".format(scoring_function, e))
            #sys.stderr.write("ERROR _call_function %s: %s\n" % (scoring_function, e))
            #traceback.print_exc()
            return None
