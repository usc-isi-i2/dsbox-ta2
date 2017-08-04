
# coding: utf-8

# In[60]:

import pandas as pd

import importlib

from os import listdir
from os.path import isfile, join


def class_for_name(module_name, class_name):
    try:
        # load the module, will raise ImportError if module cannot be loaded
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        c = getattr(m, class_name)()
        return c
    except:
        print('Could not load module: '+module_name+'.'+class_name)
        return -1

#Data: dataset to test (from the test datasets that illustrate different requirements)
#Primitive: primitive being tested. We assume it has the fit() method
#The second field returned indicates whether the primitives needs an array or not
def passTest(data, primitive):
    target = data.iloc[:,-1]
    train = data.drop(data.columns[[len(data.columns)-1]], axis=1) #drop target column (the last one)
    #test
    try:
        y_pred = primitive.fit(train,target)#.predict(train)# Not all primitives have a predict, but should have fit
        #print("PASSED: "+data.name) 
        return True, False
    except:
        #some primitives do NOT have the train for the fit method
        try:
            y_pred = primitive.fit(train)
            return True, False
        except:
            #Some primitives can only be applied to arrays, not matrix!
            try:
                for col in train.columns:
                    #print (col)
                    #Need to do the transform, otherwise exceptions may not be raised
                    y_pred = primitive.fit(train[col]).transform(train[col])
                return True, True
            except: 
                return False, False
        #print("NOT PASSED: " +data.name)
   

#Path: String with the path to the dataset folders. The system assumes to have three: clean_data, requirement_data and performance_data
#Primitive module name: string with the module name. E.g., 'sklearn.svm'
#Primitive name: string with the name of the primitive to be loaded. E.g., 'SVC'
#testPerformance: boolean that is true if you want to test the performance tests (will require more time)
def getPrimitiveRequirements(path, primitiveModuleName, primitiveName, testPerformance):
    CLEAN = path + "clean_data"
    REQ = path + "requirement_data"
    PERF = path + "performance_data"
    prim =  class_for_name(primitiveModuleName,primitiveName)
    prim.id = primitiveModuleName+"."+primitiveName
    prim.name = primitiveName
    if(prim == -1):
        print("The primitive module could not be loaded.")
        return prim
    #Clean data files: all primitives should pass these tests
    data_clean_int = pd.read_csv(CLEAN +'/int_clean_data.csv')
    data_clean_float = pd.read_csv(CLEAN +'/float_clean_data.csv')
    data_clean_int.name = "CLEAN DATA INT" 
    data_clean_float.name = "CLEAN DATA FLOAT"
    passed = (passTest(data_clean_int, prim)) and (passTest(data_clean_float, prim))
    if(not passed):
        print("The primitive "+primitiveName+" cannot execute the clean datasets. No further requirements addressed")
        return
    #Rest of the tests
    onlyfiles = [f for f in listdir(REQ) if isfile(join(REQ, f))]
    for d in onlyfiles:
        data = pd.read_csv(REQ+"/"+d)
        data.name = d
        passed,array = passTest(data, prim)
        if ("missing" in data.name) and (not passed):
            #print("Primitive cannot handle missing values")
            prim.missing = False
        if ("categorical" in data.name) and (not passed):
            #print("Primitive cannot handle string/categorical values")
            prim.categorical = False
        if ("unique" in data.name) and (not passed):
            #print("Primitive cannot handle having a column of unique values")
            prim.unique = False
        if ("negative" in data.name) and (not passed):
            #print("Primitive cannot handle negative values")
            prim.negative = False
        if(array):
            prim.isArray = True
    if(testPerformance):
        #TO DO
        print("TO DO: run the performance tests")
            
    return prim

#assumes certain variables of the JSON have been initialized.
#NON-MISSING-VALUES: The primitive cannot handle missing values
#NUMERICAL: The primitive cannot handle string/categorical values
#NOT-UNIQUE: The primitive cannot handle columns with a single value
#NON-NEGATIVE: The primitive needs to have positive values
#ARRAY: The primitive needs to be an array, not a matrix

#Will produce a JSON file that looks like:
#{
#        "class": "sklearn.linear_model.LogisticRegression",
#        "name": "LogisticRegression", 
#        "requirements": ["NUMERICAL"]
#    }
def primitiveToJSON(primitive):
    try:
        json = "{\n" + "\"class\": "+primitive.id+",\n"
        json = json + "\"name\": "+primitive.name+",\n"
        json = json + "\"requirements\":["
        #attributes are only there if false
        if hasattr(primitive, 'missing'):
            json = json + "\"NON-MISSING-VALUES\","
        if hasattr(primitive, 'categorical'):
            json = json + "\"NUMERICAL\","
        if hasattr(primitive, 'unique'):
            json = json + "\"NOT-UNIQUE\","
        if hasattr(primitive, 'negative'):
            json = json + "\"NON-NEGATIVE\","
        if hasattr(primitive, 'isArray'):
            json = json + "\"ARRAY\","
        json = json[:-1]
        json = json + "]\n}\n"
        return json
    except:
        print("Cannot serialize primitive")
        
        
#Main script        
DATADIR = "data_profiler/" #Dir with the profiling datasets
print (primitiveToJSON(getPrimitiveRequirements(DATADIR,'sklearn.svm','SVC',False)))
print (primitiveToJSON(getPrimitiveRequirements(DATADIR,'sklearn.linear_model','LogisticRegression',False)))
print (primitiveToJSON(getPrimitiveRequirements(DATADIR,'sklearn.preprocessing','LabelEncoder',False)))


# In[61]:

#Test code below
#Files with different issues: missing values, constant values, etc.
#from os import listdir
#from os.path import isfile, join
#onlyfiles = [f for f in listdir(REQ) if isfile(join(REQ, f))]
#print(onlyfiles)
#for d in onlyfiles:
#    data = pd.read_csv(DATADIR+"/requirement_data"+"/"+d)
#    data.name = d
#    passTest(data, prim)


# In[ ]:



