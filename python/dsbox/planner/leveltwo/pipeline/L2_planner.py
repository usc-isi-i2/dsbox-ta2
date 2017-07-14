from primitives.library import PrimitiveLibrary
from schema.profile import Profile
from profiler.data_profiler import Profiler

import copy
import importlib

import pandas as pd
from sklearn.model_selection import cross_val_score

class L2_Planner(object):
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
    
    """
    Function to expand the pipeline and add "glue" components
    
    :param pipeline: The input pipeline
    :param profile: The data profile
    :param index: Specifies from where to start expanding the pipeline (default 0)
    :returns: A list of expanded pipelines
    """
    def expand_pipeline(self, pipeline, profile, index=0):                
        if index >= len(pipeline):
            return None
        
        pipelines = []
        primitive = pipeline[index]
        subpipes = self._create_subpipelines(primitive, profile)
        for subpipe in subpipes:
            l2_pipeline = copy.deepcopy(pipeline)
            l2_pipeline[index:index+1] = subpipe
            nindex = index+len(subpipe)            
            cprofile = self._predict_profile(l2_pipeline, profile)            
            npipes = self.expand_pipeline(l2_pipeline, cprofile, nindex)
            if npipes:
                for npipe in npipes:
                    pipelines.append(copy.deepcopy(npipe))
            else:
                pipelines.append(l2_pipeline)

        return pipelines


    """
    Function to patch the pipeline if needed, and execute it
    
    :param pipeline: The input pipeline to patch & execute
    :param df: The data frame
    :param df_lbl: The labels/targets data frame
    :param indexcol: The column name that specifies the index in the data frames
    :param metric: The metric to compute after executing    
    :returns: A tuple containing the patched pipeline and the metric score
    """
    # TODO: Currently no patching being done
    def patch_and_execute_pipeline(self, pipeline, df, df_lbl, 
                                   indexcol="d3mIndex", metric="f1_micro"):
        cols = df.columns
        index = df.index
        
        accuracy = 0
        for primitive in pipeline: 
            # TODO: Set some default parameters ?
            exec_prim = self._instantiate_primitive(primitive, None)
            
            # TODO: Profile df here. Recheck if it is ok for primitive
            #       and patch a component here if necessary
            
            # If this is a modeling primitive
            # - we create training/test sets and check the accuracy
            if primitive.task == "Modeling":
                # Remove index columns
                df = self._prepare_data_frame(df, indexcol)
                df_lbl = self._prepare_data_frame(df_lbl, indexcol)

                # Evaluate: Get a cross validation score
                scores = cross_val_score(exec_prim, df, df_lbl.values.ravel(), 
                                         scoring=metric, cv=5)

                accuracy = scores.mean()
                break

            else:
                # If this is a non-modeling primitive, do a transformation
                if primitive.column_primitive:
                    for col in df.columns:
                        df[col] = exec_prim.fit_transform(df[col])
                else:
                    df = exec_prim.fit_transform(df)
                df = pd.DataFrame(df)
                df.columns = cols
                df.index = index

        return (pipeline, accuracy)


    def _create_subpipelines(self, primitive, profile):
        lst = []
        requirements = primitive.preconditions
        pbit = profile.profile
        
        pipeline = [primitive];
        lst.append(pipeline) 
        #print ("Profile: %s" % profile)      
        # Fulfill all requirements of the primitive if not already fulfilled
        for requirement in requirements:
            if not(requirement & pbit):
                # If requirement not fulfilled 
                print("%s requirement: %s" % 
                      (primitive.name, profile.toString(requirement)))
                glues = self.glues.getPrimitivesByEffect(requirement)
                if len(glues) == 1:
                    prim = glues[0]
                    print("-> Adding %s" % prim.name)
                    pipeline.insert(0, prim)
                elif len(glues) > 1:
                    newlst = []
                    for pipe in lst:
                        lst.remove(pipe)
                        for prim in glues:
                            cpipe = copy.deepcopy(pipe)                             
                            print("-> Adding %s" % prim.name)
                            cpipe.insert(0, prim)
                            newlst.append(cpipe)
                    lst = newlst
        return lst
    
    def _predict_profile(self, pipeline, profile):
        nprofile = copy.deepcopy(profile)
        for primitive in pipeline:
            for effect in primitive.effects:
                nprofile.profile |= effect                
                if effect >= 1<<16:
                    nprofile.profile &= ~(effect>>16)
                else:
                    nprofile.profile &= ~(effect<<16)
        #print ("Predicted profile %s" % nprofile)
        return nprofile
    
    def _instantiate_primitive(self, primitive, args):
        mod, cls = primitive.cls.rsplit('.', 1)        
        module = importlib.import_module(mod)
        PrimitiveClass = getattr(module, cls)
        if args:
            return PrimitiveClass(args)
        else:
            return PrimitiveClass()
        
    def _get_data_profile(self, df):
        df_profile_raw = Profiler(df)
        df_profile = Profile(df_profile_raw)
        return df_profile.profile
    
    def _prepare_data_frame(self, df, indexcol):
        df = pd.DataFrame(df, columns = df.columns)
        df.drop(indexcol, axis=1, inplace=True)
        return df
            
    '''
    def _get_train_test(self, df, indexcol):
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train_df = pd.DataFrame(train, columns = df.columns)
        train_df.drop(indexcol, axis=1, inplace=True)
        test_df = pd.DataFrame(test, columns = df.columns)
        test_df.drop(indexcol, axis=1, inplace=True)
        return (train_df, test_df)
    '''
  