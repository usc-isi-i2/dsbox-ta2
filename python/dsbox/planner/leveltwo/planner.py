from primitives.library import PrimitiveLibrary
from dsbox.schema.data_profile import DataProfile
from dsbox.profiler.data.data_profiler import DataProfiler

import sys
import copy
import importlib
import itertools

import pandas as pd
from sklearn.model_selection import cross_val_score

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
        
    def __init__(self, libdir):
        self.glues = PrimitiveLibrary(libdir+"/glue.json")
        self.execution_cache = {}

    
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
            
        #print "Expanding %s with index %d" % (pipeline , start_index)
        if start_index >= len(pipeline):
            # Check if there are no issues again
            npipes = self.expand_pipeline(pipeline, profile, mod_profile)
            if npipes and len(npipes) > 0:
                return npipes
            return None
       
        pipelines = []
        issues = self._get_pipeline_issues(pipeline, profile)
        #print "Issues: %s" % issues
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
                    nindex = index+len(subpipe)+len(pipeline)
                    cprofile = self._predict_profile(subpipe, profile)
                    npipes = self.expand_pipeline(l2_pipeline, profile, cprofile, nindex)
                    if npipes:
                        for npipe in npipes:
                            pipelines.append(copy.deepcopy(npipe))
                    else:
                        pipelines.append(l2_pipeline)

        if ok:
            npipelines = []
            for pipe in pipelines:
                npipelines.append(
                    self._remove_redundant_processing_primitives(pipe, profile))
            #print "Pipelines: %s " % npipelines
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
    def patch_and_execute_pipeline(self, pipeline, df, df_lbl, columns, metric="f1_micro"):
        #print "** Running Pipeline: %s" % pipeline
        
        df = copy.deepcopy(df)
        
        cols = df.columns
        
        metricvalue = 0
        cachekey = ""
        for primitive in pipeline:
            args = None 
            if primitive.task == "FeatureExtraction":
                args = {"min_df": 10, "max_features":50} # FIXME: Hack

            # TODO: Set some default parameters ?
            primitive.executable = self._instantiate_primitive(primitive, args)
            
            if not primitive.executable:
                return None
            
            cachekey += ".%s" % primitive
            if cachekey in self.execution_cache:
                #print "* Using cache for %s" % primitive
                df = self.execution_cache.get(cachekey)
                continue;
            
            #print "Executing %s" % primitive.name
            
            try:
                # TODO: Profile df here. Recheck if it is ok for primitive
                #       and patch a component here if necessary
                
                if primitive.task == "FeatureExtraction":
                    ncols = [col.format() for col in cols]
                    # TODO: Get feature extraction type (set to "text" here)
                    textcols = self._columns_of_type(columns, "text")
                    for col in textcols:
                        nvals = primitive.executable.fit_transform(df[col])
                        fcols = [(col.format() + "_" + feature) 
                                 for feature in primitive.executable.get_feature_names()]
                        newdf = pd.DataFrame(nvals.toarray(), 
                                             columns=fcols)
                        del df[col]
                        ncols = ncols + fcols
                        ncols.remove(col)               
                        df = pd.concat([df, newdf], axis=1)
                        df.columns=ncols
                    cols = df.columns
                    self.execution_cache[cachekey] = df
                    
                # If this is a modeling primitive
                # - we create training/test sets and check the metricvalue
                elif primitive.task == "Modeling":
                    # Fit the Model
                    primitive.executable.fit(df, df_lbl.values.ravel())

                    # Evaluate: Get a cross validation score for the metric
                    scores = cross_val_score(primitive.executable, df, df_lbl.values.ravel(), 
                                             scoring=metric, cv=5)
    
                    metricvalue = scores.mean()
                    break
    
                else:
                    # If this is a non-modeling primitive, do a transformation
                    if primitive.column_primitive:
                        for col in df.columns:
                            df[col] = primitive.executable.fit_transform(df[col])
                    else:
                        df = primitive.executable.fit_transform(df)
    
                    df = pd.DataFrame(df)
                    df.columns = cols
                    self.execution_cache[cachekey] = df
            
            except Exception, e:
                #print "ERROR: %s" % e
                return None

        return (pipeline, metricvalue)
    
    def _remove_redundant_processing_primitives(self, pipeline, profile):
        curpipe = copy.copy(pipeline)
        length = len(curpipe)-1
        index = 0
        #print "Checking redundancy for %s" % pipeline
        while index <= length:
            prim = curpipe.pop(index)
            #print "%s / %s: %s" % (index, length, prim)            
            if prim.task == "PreProcessing":
                issues = self._get_pipeline_issues(curpipe, profile)
                ok = True
                for issue in issues:
                    if len(issue):
                        ok = False 
                if ok:
                    #print "Reduction achieved"
                    # Otherwise reduce the length (and don't increment index)
                    length = length - 1
                    continue

            curpipe[index:index] = [prim]
            #print curpipe                    
            index += 1
    
        #print "Returning %s" % curpipe
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
        #print "Profiles: %s\nRequirements: %s" % (profiles, requirements)
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
        for index in range(0, len(pipeline)-1):
            primitive = pipeline[index]
            nprofile = copy.deepcopy(curprofile)
            for effect in primitive.effects.keys():
                nprofile.profile[effect] = primitive.effects[effect]
            profiles.append(nprofile)
            curprofile = nprofile
        return profiles
    
    def _get_pipeline_requirements(self, pipeline):
        requirements = [];
        effects = [];
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
                        #lst.remove(pipe)
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
    
    def _instantiate_primitive(self, primitive, args):
        mod, cls = primitive.cls.rsplit('.', 1)
        try:        
            module = importlib.import_module(mod)
            PrimitiveClass = getattr(module, cls)
            if args:
                return PrimitiveClass(**args)
            else:
                return PrimitiveClass()
        except Exception, e:
            #print "ERROR: %s" % e
            return None
        
    def _get_data_profile(self, df):
        df_profile_raw = Profiler(df)
        df_profile = Profile(df_profile_raw)
        return df_profile.profile
    
    def _columns_of_type(self, columns, type):
        cols = []
        for c in columns:
            if c['varType'] == type:
                cols.append(c['varName'])
        return cols 
    
    '''
    def _get_train_test(self, df, indexcol):
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train_df = pd.DataFrame(train, columns = df.columns)
        train_df.drop(indexcol, axis=1, inplace=True)
        test_df = pd.DataFrame(test, columns = df.columns)
        test_df.drop(indexcol, axis=1, inplace=True)
        return (train_df, test_df)
    '''
  
