from .primitives.library import PrimitiveLibrary
from dsbox.schema.data_profile import DataProfile
from dsbox.profiler.data.data_profiler import DataProfiler
from dsbox.schema import TaskType

import os
import sys
import copy
import itertools
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer

TIMEOUT = 300  # Time out primitives running for more than 5 minutes


class LevelTwoPlanner(object):
    """
    The Level-2 DSBox Planner.

    The function "expand_pipeline" is used to expand a Level 1 Pipeline (which
    contains modeling and possibly featurization steps) into a "Level 2 pipeline"
    that can be executed by making sure that the provided data satisfies preconditions
    of steps. This is done by inserting "Glue" or "PreProcessing" primitives into
    the pipeline.

    The function "patch_and_execute_pipeline" is used to execute a Level 2 Pipeline
    and while executing ensure that the intermediate data that is produced does indeed
    match the data profile that was expected in the "expand_pipeline" function. If it
    does not match, then some more "glue" components are patched in to ensure compliance
    with primitive preconditions. The result of this function is a list of
    (patched_pipeline, metric_value) tuples. The metric_value is the value of the type of
    metric that is passed to the function. Examples are "accuracy", "f1_macro", etc.
    """

    def __init__(self, libdir, helper):
        self.glues = PrimitiveLibrary(libdir + os.sep + "glue.json")
        self.execution_cache = {}
        self.primitive_cache = {}
        self.helper = helper

    """
    Function to expand the pipeline and add "glue" primitives

    :param pipeline: The input pipeline
    :param profile: The data profile
    :param mod_profile: The modified data profile
    :param index: Specifies from where to start expanding the pipeline (default 0)
    :returns: A list of expanded pipelines
    """

    def expand_pipeline(self, pipeline, profile, mod_profile=None, start_index=0):
        if not mod_profile:
            mod_profile = profile

        #print("Expanding %s with index %d" % (pipeline , start_index))
        if start_index >= len(pipeline):
            # Check if there are no issues again
            npipes = self.expand_pipeline(pipeline, profile, mod_profile)
            if npipes and len(npipes) > 0:
                return npipes
            return None

        pipelines = []
        issues = self._get_pipeline_issues(pipeline, profile)
        #print("Issues: %s" % issues)
        ok = True
        for index in range(start_index, len(pipeline)):
            primitive = pipeline[index]
            issue = issues[index]
            if len(issue) > 0:
                ok = False
                # There are some unresolved issues with this primitive
                # Try to resolve it
                subpipes = self._create_subpipelines(primitive, issue)
                for subpipe in subpipes:
                    ok = True
                    l2_pipeline = copy.deepcopy(pipeline)
                    l2_pipeline[index:index] = subpipe
                    nindex = index + len(subpipe) + len(pipeline)
                    cprofile = self._predict_profile(subpipe, profile)
                    npipes = self.expand_pipeline(
                        l2_pipeline, profile, cprofile, nindex)
                    if npipes:
                        for npipe in npipes:
                            pipelines.append(copy.deepcopy(npipe))
                    else:
                        pipelines.append(l2_pipeline)

        if ok:
            if len(pipelines) == 0:
                # No additions, use existing pipeline
                pipelines.append(copy.deepcopy(pipeline))
            npipelines = []
            for pipe in pipelines:
                npipelines.append(
                    self._remove_redundant_processing_primitives(pipe, profile))
            # print "Pipelines: %s " % npipelines
            return self._remove_duplicate_pipelines(npipelines)
        else:
            return None

    """
    Function to patch the pipeline if needed, and execute it

    :param pipeline: The input pipeline to patch & execute
    :param df: The data frame
    :param df_lbl: The labels/targets data frame
    :param metric: The metric to compute after executing
    :returns: A tuple containing the patched pipeline and the metric score
    """
    # TODO: Currently no patching being done

    def patch_and_execute_pipeline(self, pipeline, df, df_lbl, columns, task_type, metric, metric_function):
        print("** Running Pipeline: %s" % pipeline)

        df = copy.deepcopy(df)
        cols = df.columns

        metricvalue = 0
        cachekey = ""
        for primitive in pipeline:
            args = []
            kwargs = {}

            if primitive.getInitKeywordArgs():
                primitive.init_kwargs = self.helper._process_kwargs(
                    primitive.getInitKeywordArgs(), task_type, metric)

            if primitive.getInitArgs():
                primitive.init_args = self.helper._process_args(
                    primitive.getInitArgs(), task_type, metric)

            cachekey += ".%s" % primitive
            if cachekey in self.execution_cache:
                # print "* Using cache for %s" % primitive
                df = self.execution_cache.get(cachekey)
                primitive.executables = self.primitive_cache.get(cachekey)
                continue

            try:
                print("Executing %s" % primitive.name)

                # Re-profile intermediate data here.
                # TODO: Recheck if it is ok for the primitive's preconditions
                #       and patch pipeline if necessary
                cur_profile = DataProfile(df)

                if primitive.task == "FeatureExtraction":
                    # Featurisation Primitive
                    df = self.helper.featurise(df, primitive, timeout=TIMEOUT)
                    cols = df.columns
                    self.execution_cache[cachekey] = df
                    self.primitive_cache[cachekey] = primitive.executables

                elif primitive.task == "Modeling":
                    # Modeling Primitive
                    # Evaluate: Get a cross validation score for the metric
                    metricvalue = self.helper.cross_validation_score(primitive, df, df_lbl,
                                                                     metric, metric_function, 5, timeout=TIMEOUT)
                    if not metricvalue:
                        return None

                    self.primitive_cache[cachekey] = primitive.executables
                    break

                else:
                    # Glue primitive
                    df = self.helper.execute_primitive(
                        primitive, df, df_lbl, cur_profile, timeout=TIMEOUT)
                    self.execution_cache[cachekey] = df
                    self.primitive_cache[cachekey] = primitive.executables

            except Exception as e:
                sys.stderr.write(
                    "ERROR patch_and_execute_pipeline(%s) : %s\n" % (pipeline, e))
                traceback.print_exc()
                return None

        return (pipeline, metricvalue)

    def _remove_redundant_processing_primitives(self, pipeline, profile):
        curpipe = copy.copy(pipeline)
        length = len(curpipe) - 1
        index = 0
        # print "Checking redundancy for %s" % pipeline
        while index <= length:
            prim = curpipe.pop(index)
            # print "%s / %s: %s" % (index, length, prim)
            if prim.task == "PreProcessing":
                issues = self._get_pipeline_issues(curpipe, profile)
                ok = True
                for issue in issues:
                    if len(issue):
                        ok = False
                if ok:
                    # print "Reduction achieved"
                    # Otherwise reduce the length (and don't increment index)
                    length = length - 1
                    continue

            curpipe[index:index] = [prim]
            # print curpipe
            index += 1

        # print "Returning %s" % curpipe
        return curpipe

    def _remove_duplicate_pipelines(self, pipelines):
        pipes = []
        pipehash = {}
        for pipeline in pipelines:
            hash = str(pipeline)
            pipehash[hash] = pipeline
            pipes.append(hash)
        pipeset = set(pipes)
        pipes = []
        for pipe in pipeset:
            pipes.append(pipehash[pipe])
        return pipes

    def _get_pipeline_issues(self, pipeline, profile):
        unmet_requirements = []
        profiles = self._get_predicted_data_profiles(pipeline, profile)
        requirements = self._get_pipeline_requirements(pipeline)
        # print "Profiles: %s\nRequirements: %s" % (profiles, requirements)
        for index in range(0, len(pipeline)):
            unmet = {}
            prim_prec = requirements[index]
            profile = profiles[index]
            for requirement in prim_prec.keys():
                reqvalue = prim_prec[requirement]
                if reqvalue != profile.profile.get(requirement, None):
                    unmet[requirement] = reqvalue
            unmet_requirements.append(unmet)
        return unmet_requirements

    def _get_predicted_data_profiles(self, pipeline, profile):
        curprofile = copy.deepcopy(profile)
        profiles = [curprofile]
        for index in range(0, len(pipeline) - 1):
            primitive = pipeline[index]
            nprofile = copy.deepcopy(curprofile)
            for effect in primitive.effects.keys():
                nprofile.profile[effect] = primitive.effects[effect]
            profiles.append(nprofile)
            curprofile = nprofile
        return profiles

    def _get_pipeline_requirements(self, pipeline):
        requirements = []
        effects = []
        for index in range(0, len(pipeline)):
            primitive = pipeline[index]
            prim_requirements = {}
            for prec in primitive.preconditions.keys():
                prim_requirements[prec] = primitive.preconditions[prec]

            if index > 0:
                # Make effects of previous primitives satisfy any preconditions
                for oldindex in range(0, index):
                    last_prim = pipeline[oldindex]
                    for effect in last_prim.effects.keys():
                        prim_requirements[effect] = last_prim.effects[effect]

            requirements.append(prim_requirements)
        return requirements

    def _create_subpipelines(self, primitive, prim_requirements):
        mainlst = []

        requirement_permutations = list(
            itertools.permutations(prim_requirements))

        for requirements in requirement_permutations:
            #print("%s requirement: %s" % (primitive.name, requirements))
            xpipe = []
            lst = [xpipe]
            # Fulfill all requirements of the primitive
            for requirement in requirements:
                reqvalue = prim_requirements[requirement]
                glues = self.glues.getPrimitivesByEffect(requirement, reqvalue)
                if len(glues) == 1:
                    prim = glues[0]
                    #print("-> Adding one %s" % prim.name)
                    xpipe.insert(0, prim)
                elif len(glues) > 1:
                    newlst = []
                    for pipe in lst:
                        # lst.remove(pipe)
                        for prim in glues:
                            cpipe = copy.deepcopy(pipe)
                            #print("-> Adding %s" % prim.name)
                            cpipe.insert(0, prim)
                            newlst.append(cpipe)
                    lst = newlst
            mainlst += lst

        return mainlst

    def _predict_profile(self, pipeline, profile):
        curprofile = copy.deepcopy(profile)
        for primitive in pipeline:
            for effect in primitive.effects.keys():
                curprofile.profile[effect] = primitive.effects[effect]
        #print ("Predicted profile %s" % curprofile)
        return curprofile

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
                return LinearRegression
            else:
                raise Exception("Not yet implemented: Arg specification ESTIMATOR task type: {}"
                                .format(task_type))
        else:
            raise Exception(
                "Unkown Arg specification: {}".format(arg_specification))
