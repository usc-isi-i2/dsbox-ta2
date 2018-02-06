import os
import sys
import copy
import traceback

import multiprocessing
from multiprocessing import Pool, Queue, Lock
from dsbox.schema.data_profile import DataProfile
from dsbox.planner.common.pipeline import Pipeline, PipelineExecutionResult

TIMEOUT = 600  # Time out primitives running for more than 10 minutes

class DummyAsyncResult(object):
    def __init__(self, func, args):
        self.func = func
        self.args = args
    def get(self, timeout=0):
        return self.func(*self.args)

class ResourceManager(object):
    EXECUTION_POOL = None

    def __init__(self, helper, numcpus=0):
        self.helper = helper
        if ResourceManager.EXECUTION_POOL is None:
            if numcpus == 0:
                numcpus = multiprocessing.cpu_count()
            ResourceManager.EXECUTION_POOL = Pool(numcpus)
        #self.primitive_queue = Queue()
        #self.lock = Lock()

        self.primitive_cache = {}
        self.execution_cache = {}

        self.use_apply_async = True

    def execute_pipelines(self, pipelines, df, df_lbl):
        pipeline_refs = []
        exec_pipelines = []
        for pipeline in pipelines:
            pipeline_refs.append(self.execute_pipeline(pipeline, df, df_lbl))
        for pref in pipeline_refs:
            if pref is not None:
                (exref, exec_pipeline, primitive, cachekey, newdf, newdf_lbl) = pref
                if exref is not None:
                    try:
                        retvalue = exref.get(timeout=TIMEOUT)
                        if retvalue is not None:
                            # Get model predictions and metric values
                            (predictions, metric_values) = retvalue
                            if metric_values and len(metric_values) > 0:
                                print("Got results from {}".format(primitive))
                                # Create a model for the primitive (Not parallel so we can store the model for evaluation)
                                self.helper.create_primitive_model(primitive, newdf, newdf_lbl)
                                # Store the execution result
                                exec_pipeline.planner_result = PipelineExecutionResult(predictions, metric_values)
                                # Cache the primitive instance
                                self.primitive_cache[cachekey] = (primitive.executables, primitive.unified_interface)
                                # Add to the list of executable pipelines
                                exec_pipelines.append(exec_pipeline)
                    except Exception as e:
                        traceback.print_exc()
                        sys.stderr.write("ERROR execute_pipelines(%s) : %s\n" % (pipeline, e))
        return exec_pipelines

    def execute_pipeline(self, pipeline, df, df_lbl):
        print("** Running Pipeline: %s" % pipeline)
        sys.stdout.flush()

        exec_pipeline = pipeline.clone(idcopy=True)

        # TODO: Check for ramifications
        pipeline.primitives = exec_pipeline.primitives
        cols = df.columns

        cachekey = ""

        for primitive in exec_pipeline.primitives:
            # Mark the pipeline that the primitive is part of
            # - Used to notify waiting threads of execution changes
            primitive.pipeline = pipeline
            cachekey = "%s.%s" % (cachekey, primitive.cls)
            if cachekey in self.execution_cache:
                # print ("* Using cache for %s" % primitive)
                df = self.execution_cache.get(cachekey)
                (primitive.executables, primitive.unified_interface) = self.primitive_cache.get(cachekey)
                continue

            try:
                if df is None:
                    return None

                # FIXME: HACK !!!!
                # Some primitives are not fork-safe. Should not use multiprocessor.Pool after using these components
                if 'ImageFeature' in primitive.cls or 'BBNAudioPrimitiveWrapper' in primitive.cls:
                    self.use_apply_async = False

                if primitive.task == "FeatureExtraction":
                    print("Executing %s" % primitive.name)
                    sys.stdout.flush()

                    # Featurisation Primitive
                    df = self.helper.featurise(primitive, copy.copy(df), timeout=TIMEOUT)
                    cols = df.columns
                    self.execution_cache[cachekey] = df
                    self.primitive_cache[cachekey] = (primitive.executables, primitive.unified_interface)

                elif primitive.task == "Modeling":
                    # Modeling Primitive

                    if self.use_apply_async:
                        # Evaluate: Get a cross validation score for the metric
                        # Return reference to parallel process
                        exref = ResourceManager.EXECUTION_POOL.apply_async(
                            self.helper.cross_validation_score,
                            args=(primitive, df, df_lbl, 10))
                    else:
                        exref = DummyAsyncResult(
                            self.helper.cross_validation_score,
                            args=(primitive, df, df_lbl, 10))
                    return (exref, exec_pipeline, primitive, cachekey, df, df_lbl)

                else:
                    print("Executing %s" % primitive.name)
                    sys.stdout.flush()

                    # Re-profile intermediate data here.
                    # TODO: Recheck if it is ok for the primitive's preconditions
                    #       and patch pipeline if necessary
                    cur_profile = DataProfile(df)

                    # Glue primitive
                    df = self.helper.execute_primitive(
                        primitive, copy.copy(df), df_lbl, cur_profile, timeout=TIMEOUT)
                    self.execution_cache[cachekey] = df
                    self.primitive_cache[cachekey] = (primitive.executables, primitive.unified_interface)

            except Exception as e:
                sys.stderr.write(
                    "ERROR execute_pipeline(%s) : %s\n" % (exec_pipeline, e))
                traceback.print_exc()
                exec_pipeline.finished = True
                return None

        pipeline.finished = True
        return exec_pipeline


    def primitive_executed(self, primitive, status):
        pass

    def execute_primitive(self, primitive, callback_fn):
        pass

    '''
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['execution_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
    '''
