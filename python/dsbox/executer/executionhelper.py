import os
import sys
import json
import time
import os.path
import tempfile
import numpy as np
import pandas as pd
import contextlib

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

from dsbox.planner.common.pipeline import CrossValidationStat

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
                    executable = PrimitiveClass(hyperparams=primitive.getHyperparams())
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


    def execute_primitive_remote(self, primitive, df, df_lbl, cur_profile=None):
        '''Remote version execute_primitive'''
        pd = self.execute_primitive(primitive, df, df_lbl, cur_profile)
        return (pd, primitive.executables, primitive.unified_interface)

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
                print("ERROR: execute_primitive {}: {}\n".format(primitive.name, e))
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
                        if df[col].dtype == object:
                            df[col] = df[col].fillna('')
                        else:
                            df[col] = df[col].fillna(0)
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
    def cross_validation_score(self, primitive, X, y, cv=4, seed=0):
        print("Executing %s" % primitive.name)
        sys.stdout.flush()

        metric_values = {}  # Dict[str, float]
        stat = CrossValidationStat()

        # Redirect stderr to an error file
        #  Directly assigning stderr to tempfile.TemporaryFile cause printing str to fail
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, primitive.name), 'w') as errorfile:
                with contextlib.redirect_stderr(errorfile):

                    primitive.start_time = time.time()

                    # TODO: Should use same random_state for comparison across algorithms
                    
                    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)#int(time.time()))
                    
                    tcols = [self.data_manager.target_columns[0]['colName']]
                    yPredictions = None
                    num = 0.0
                    for k, (train, test) in enumerate(kf.split(X, y)):
                        executable = self.instantiate_primitive(primitive)
                        if executable is None:
                            primitive.finished = True
                            return (None, None, None)

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

                            # TODO: Training metrics for each fold

                            # Test metrics for each fold
                            for i in range(0, len(self.problem.metrics)):
                                metric = self.problem.metrics[i]
                                fn = self.problem.metric_functions[i]
                                fold_metric_val = self._call_function(fn, y.loc[ypredDF.index], ypredDF)
                                stat.add_fold_metric(metric, fold_metric_val)

                        except Exception as e:
                            sys.stderr.write("ERROR: cross_validation {}: {}\n".format(primitive.name, e))
                            # traceback.print_exc(e)

        if num == 0:
            return (None, None, None)

        yPredictions = yPredictions.sort_index()

        #print ("Trained on {} samples, Tested on {} samples".format(len(train), len(ypred)))
        for i in range(0, len(self.problem.metrics)):
            metric = self.problem.metrics[i]
            fn = self.problem.metric_functions[i]
            metric_val = self._call_function(fn, y, yPredictions)
            if metric_val is None:
                return (None, None, None)
            metric_values[metric.name] = metric_val

        primitive.end_time = time.time()
        primitive.progress = 1.0
        primitive.finished = True
        primitive.pipeline.notifyChanges()

        print('metric values = {}'.format(metric_values))
        for metric in self.problem.metrics:
            print('Cross valiation standard error for meric {} = {}'.format(
                metric, stat.get_standard_error(metric)))

        #print ("Returning {}".format(metric_values))
        return (yPredictions, metric_values, stat)


    def create_primitive_model_remote(self, primitive, X, y):
        '''Remote version of create_primitive_model'''
        self.create_primitive_model( primitive, X, y)
        return (primitive.executables, primitive.unified_interface)



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

    def featurise_remote(self, primitive, df):
        '''Use this method if running in subprocess of remotely'''
        pd = self.featurise(primitive, df)

        # Return executable as well
        return (pd, primitive.executables, primitive.unified_interface)

    @stopit.threading_timeoutable()
    def featurise(self, primitive, df):
        primitive.start_time = time.time()

        persistent = primitive.is_persistent
        ncols = [col.format() for col in df.columns]
        featurecols = self.raw_data_columns(self.data_manager.input_columns)
        indices = df.index

        # TO DO : TEST FEATURIZATION FOR DATA WITHOUT MEDIA TYPE
        if False and (self.data_manager.media_type == VariableFileType.NONE or self.data_manager.media_type is None):
            print('------ Featurization with primitive --------- ', primitive.cls.split('.')[-1])
            executable = self.instantiate_primitive(primitive)
            if executable is None:
                primitive.finished = True
                return None
            
            executable.set_training_data(inputs=df.values, outputs=[])
            executable.fit()
            call_result = executable.produce(inputs=df.values)
            fcols = [(primitive.cls.split('.')[-1].format() + "_" + str(index)) for index in range(0, call_result.value.shape[1])]
            newdf = pd.DataFrame(call_result.value, columns=fcols, index=df.index)
            ncols = ncols + fcols
            df = pd.concat([df, newdf], axis=1)
            df.columns = ncols

        else:
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

                  #nvals = executable.fit_transform(df[col])
                  executable.set_training_data(inputs=df[col].values, outputs=[])
                  if persistent:
                      executable.fit()
                      call_result = executable.produce(inputs=df[col].values)

                  #fcols = [(col.format() + "_" + feature) for feature in executable.get_feature_names()]
                  val = call_result.value.todense() if isinstance(call_result.value, scipy.sparse.csr.csr_matrix) else call_result.value
                  fcols = [(col.format() + "_" + str(index)) for index in range(0, val.shape[1])]
                  newdf = pd.DataFrame(val, columns=fcols, index=df.index)
                  del df[col]
                  ncols = ncols + fcols
                  ncols.remove(col)
                  df = pd.concat([df, newdf], axis=1)
                  df.columns = ncols
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
                  if executable is None:
                      primitive.finished = True
                  executable = self.instantiate_primitive(primitive)

                  # df[col] is (1d array, sampling)rate)
                  call_result = executable.produce(inputs=pd.DataFrame(df[col]))

                  # call_result.value is list of length df.shape[0]
                  # Where each element is a list of 2d ndarrays
                  # The length of the nested list is depended on the sound clip length
                  # We should turn each element in the nested list into a new instance,
                  # but now just average over all nested elements.

                  features = call_result.value
                  rows = []
                  for row_list in features:
                      total = np.zeros((row_list[0].size,))
                      for elt in row_list:
                          total += elt.flatten()
                      total /= len(row_list)
                      rows.append(total)
                  col_names = ['{}_{}'.format(col.format(), i) for i in range(total.size)]

                  newdf = pd.DataFrame(rows, index=df.index, columns=col_names)

                  # FIXME: Need to be more general
                  df.drop(['filename', 'start', 'end'], axis=1, inplace=True)
                  df = pd.concat([df, newdf], axis=1)

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
                    call_result = executable.produce(df[col])
                else:
                    executable.fit()
                    call_result = executable.produce(inputs=df[col].values)

                val = call_result.value.todense() if isinstance(call_result.value, scipy.sparse.csr.csr_matrix) else call_result.value
                #fcols = [(col.format() + "_" + feature) for feature in executable.get_feature_names()]
                fcols = [(col.format() + "_" + str(index)) for index in range(0, val.shape[1])]
                newdf = pd.DataFrame(val, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns=ncols
            elif self.data_manager.media_type == VariableFileType.TIMESERIES:
                call_result = executable.produce(inputs=df[col].values)
                features = call_result.value
                fcols = [(col.format() + "_" + str(index)) for index in range(0, features.shape[1])]
                newdf = pd.DataFrame(call_result.value, columns=fcols, index=df.index)
                del df[col]
                ncols = ncols + fcols
                ncols.remove(col)
                df = pd.concat([df, newdf], axis=1)
                df.columns = ncols
            elif self.data_manager.media_type == VariableFileType.IMAGE:
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
                call_result = executable.produce(inputs=pd.DataFrame(df[col]))
                features = call_result.value
                rows = []
                for row_list in features:
                    total = np.zeros((row_list[0].size,))
                    for elt in row_list:
                        total += elt.flatten()
                    total /= len(row_list)
                    rows.append(total)
                col_names = ['{}_{}'.format(col.format(), i) for i in range(total.size)]

                newdf = pd.DataFrame(rows, index=df.index, columns=col_names)

                # FIXME: Need to be more general
                df.drop(['filename', 'start', 'end'], axis=1, inplace=True)
                df = pd.concat([df, newdf], axis=1)

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
        #statements.append("\ntestdata = data_manager.input_data")

        statements.append("\nhp = ExecutionHelper(problem, data_manager)")
        index = 1

        ensembling = pipeline.ensemble is not None
        n_pipelines = len(pipeline.ensemble.pipelines) if ensembling else 1
        ens_pipeline = pipeline

        if ensembling:
            [low_pred, hi_pred] = pipeline.ensemble.prediction_range
            statements.append("results = []")
            median = pipeline.ensemble.median

        variable_cache = {}
        varindex = 0
        for pipe_i in range(n_pipelines):
            statements.append("\ntestdata_0 = data_manager.input_data")
            if ensembling:
                pipeline = ens_pipeline.ensemble.pipelines[pipe_i]

            cachekey = ""
            for primitive in pipeline.primitives:
                varid = "testdata_%s" % str(varindex)
                newvarid = "testdata_%s" % str(varindex+1)

                if cachekey in variable_cache:
                    # print ("* Using cache for %s" % primitive)
                    varid = variable_cache.get(cachekey)
                else:
                    variable_cache[cachekey] = varid

                cachekey = "%s.%s" % (cachekey, primitive.cls)
                if cachekey in variable_cache:
                    newvarid = variable_cache.get(cachekey)
                    continue

                variable_cache[cachekey] = newvarid
                varindex += 1

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
                    primfile = "%s%s%s" % (tmp_dir, os.sep, primfilename)
                    statements.append("primfile = temp_storage_root + '%s%s'" % (os.sep, primfilename))
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
                    #statements.append("if not os.path.exists(results_root):")
                    #statements.append("    os.makedirs(results_root)")
                    target_column = self.data_manager.target_columns[0]['colName']

                    if primitive.unified_interface:
                        statements.append("result = pandas.DataFrame(%s.executables.produce(inputs=%s).value, index=%s.index, columns=['%s'])" %
                            (primid, varid, varid, target_column))
                    else:
                        statements.append("result = pandas.DataFrame(%s.executables.predict(%s), index=%s.index, columns=['%s'])" %
                            (primid, varid, varid, target_column))

                    if ensembling:
                        statements.append("results.append(result)")

                else:
                    if primitive.task == "PreProcessing":
                        statements.append("%s = hp.test_execute_primitive(%s, %s)" % (newvarid, primid, varid))
                    elif primitive.task == "FeatureExtraction":
                        statements.append("%s = hp.test_featurise(%s, %s)" % (newvarid, primid, varid))

                index += 1

        # Write results
        statements.append("\nprint('\\nStoring results in %s' % predictions_file)")
        statements.append("if not os.path.exists(results_root):")
        statements.append("    os.makedirs(results_root)")
        statements.append("")

        # transform categorical labels?
        if not ensembling:
            statements.append("result.to_csv(predictions_file, index_label='%s')" % self.data_manager.index_column)
        else:
            statements.append("results_np = numpy.array([df.values for df in results])")
            # ONLY to how many pipelines have executed
            weights_string = ', '.join([str(w) for w in ens_pipeline.ensemble.pipeline_weights[:pipe_i+1]])
            statements.append("weights_np = numpy.array([%s]).astype(numpy.int32)" % weights_string)
            #statements.append("weighted_total = numpy.array([df*const for df, const in zip(results_np, weights_np)])")
            #statements.append("average_pred = numpy.sum(weighted_total, axis = 0)/numpy.sum(weights_np)")

            if median:
                statements.append("results_np = numpy.repeat(results_np, repeats = weights_np, axis = 0)")
                statements.append("ens_pred = numpy.median(results_np, axis = 0)")
            else:
                statements.append("weight_mask = numpy.multiply(weights_np[:,numpy.newaxis, numpy.newaxis], numpy.logical_and(results_np >= %s, results_np <= %s))" % (low_pred, hi_pred))
                statements.append("ens_pred = numpy.average(results_np, axis = 0, weights = weight_mask)")

            if ens_pipeline.ensemble.discrete_metric:
                statements.append("ens_pred = numpy.rint(ens_pred)")
            statements.append("result = pandas.DataFrame(ens_pred, index=testdata_0.index, columns=['%s'])" % self.data_manager.target_columns[0]['colName'])
            statements.append("result.to_csv(predictions_file, index_label='%s')" % self.data_manager.index_column)
            # ~ timeout check

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
