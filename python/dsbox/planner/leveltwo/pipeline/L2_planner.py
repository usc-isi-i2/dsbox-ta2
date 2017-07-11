from primitives.library import PrimitiveLibrary
from schema.profile import Profile
from profiler.data_profiler import Profiler

import copy
import importlib

import pandas as pd
from sklearn.model_selection import train_test_split

class L2_Planner(object):
    def __init__(self, libdir):
        self.glues = PrimitiveLibrary(libdir+"/glue.json")
        
    def expand_pipeline(self, l1_pipeline, profile, index=0):
        if index >= len(l1_pipeline):
            return None
        
        pipelines = []
        primitive = l1_pipeline[index]
        subpipes = self.create_subpipelines(primitive, profile)
        for subpipe in subpipes:
            l2_pipeline = copy.deepcopy(l1_pipeline)
            l2_pipeline[index:index+1] = subpipe
            nindex = index+len(subpipe)+1
            cprofile = self.predict_profile(l2_pipeline, profile)
            npipes = self.expand_pipeline(l2_pipeline, cprofile, nindex)
            if npipes:
                for npipe in npipes:
                    cpipe2 = copy.deepcopy(l2_pipeline)
                    cpipe2[nindex:nindex+1] = npipe
                    pipelines.append(l2_pipeline)
            else:
                pipelines.append(l2_pipeline)

        return pipelines
        
    def create_subpipelines(self, primitive, profile):
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
    
    
    def predict_profile(self, pipeline, profile):
        nprofile = copy.deepcopy(profile)
        for primitive in pipeline:
            for effect in primitive.effects:
                if effect >= 1<<16:
                    nprofile.profile &= ~(effect>>16)
                else:
                    nprofile.profile &= ~(effect<<16)
        return nprofile
    
    # TODO: Currently no patching being done
    def patch_and_execute_pipeline(self, l2_pipeline, df, df_lbl, indexcol):
        cols = df.columns
        index = df.index
        
        accuracy = 0
        for primitive in l2_pipeline: 
            # TODO: Set parameters
            exec_prim = self.instantiate_primitive(primitive, None)
            
            # TODO: Profile df here. Recheck if it is ok for primitive
            
            #print "Executing %s" % primitive
            
            # If this is a modeling primitive
            # - we create training/test sets and check the accuracy
            if primitive.task == "Modeling":
                # Split Training and Test Data
                train, test = self.get_train_test(df, indexcol)
                train_lbl, test_lbl = self.get_train_test(df_lbl, indexcol)     
                accuracy = exec_prim.fit(train, train_lbl.values.ravel()).score(test, test_lbl)
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

        return (l2_pipeline, accuracy)

    
    def instantiate_primitive(self, primitive, args):
        mod, cls = primitive.cls.rsplit('.', 1)        
        module = importlib.import_module(mod)
        PrimitiveClass = getattr(module, cls)
        if args:
            return PrimitiveClass(args)
        else:
            return PrimitiveClass()
        
    def get_data_profile(self, df):
        df_profile_raw = Profiler(df)
        df_profile = Profile(df_profile_raw)
        return df_profile.profile
    
    
    def get_train_test(self, df, indexcol):
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train_df = pd.DataFrame(train, columns = df.columns)
        train_df.drop(indexcol, axis=1, inplace=True)
        test_df = pd.DataFrame(test, columns = df.columns)
        test_df.drop(indexcol, axis=1, inplace=True)
        return (train_df, test_df)
        
    '''
    def add_glue(self, df):
        # Get current profile
        df_profile_raw = profiler.Profiler(df)
        df_profile = schema.Profile(df_profile_raw)
        print("Current Profile:")
        print(df_profile.profile)
                
        # Fulfill "Numerical Values" Requirement
        if not(df_profile.profile & schema.Profile.NUMERICAL):
            glue = preprocessing.LabelEncoder()
            for col, profile in df_profile.columns.items():
                if not(profile & schema.Profile.NUMERICAL):
                    df[col] = glue.fit_transform(df[col])

            # Get current profile
            df_profile_raw = profiler.Profiler(df)
            df_profile = schema.Profile(df_profile_raw)
            print("Profile after fulfilling 'Numerical' requirement: ")
            print(df_profile.profile)
        
        # Fulfill "No Missing Values" Requirement
        if (df_profile.profile & schema.Profile.MISSING_VALUES):
            glue = preprocessing.Imputer()
            df1 = pd.DataFrame(glue.fit_transform(df))
            df1.columns = df.columns
            df1.index = df.index
            df = df1
    
            # Get current profile
            df_profile_raw = profiler.Profiler(df)
            df_profile = schema.Profile(df_profile_raw)
            print("Profile after fulfilling 'No Missing Values' requirement: ")
            print(df_profile.profile)
            
        return df
    
    '''