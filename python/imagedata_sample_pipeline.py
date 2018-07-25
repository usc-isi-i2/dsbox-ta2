# This is a sample pipeline file to make a prediction on the image 22_handgeometry_dataset
import sys
import os
import pprint
from d3m.container.dataset import D3MDatasetLoader, Dataset, CSVLoader
from common_primitives.denormalize import DenormalizePrimitive, Hyperparams as hyper_DE
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive, Hyperparams as hyper_DD
#from dsbox.datapreprocessing.featurizer.image.dataframe_to_tensor import DataFrameToTensor
#from dsbox.datapreprocessing.featurizer.image.net_image_feature import Vgg16ImageFeature
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive, Hyperparams as hyper_EX


context_input = 'PRETRAINING'
sample_pipeline = Pipeline(context = context_input)

h1 = hyper_DE.defaults()
h2 =hyper_DD.defaults()
h3 = {}
h4 = {'layer_index':3}
h5 = hyper_EX.defaults()
h6 = {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target','https://metadata.datadrivendiscovery.org/types/SuggestedTarget',), 'use_columns': (), 'exclude_columns': ()}
primitive_0 = DenormalizePrimitive(hyperparams = h1)
primitive_1 = DatasetToDataFramePrimitive(hyperparams = h2)
#primitive_2 = DataFrameToTensor(hyperparams = h3)
#primitive_3 = Vgg16ImageFeature(hyperparams = h4)
primitive_4 = ExtractColumnsBySemanticTypesPrimitive(hyperparams = h6)

step_0 = PrimitiveStep(primitive_0.metadata.query())
step_1 = PrimitiveStep(primitive_1.metadata.query())
#step_2 = PrimitiveStep(primitive_2.metadata.query())
#step_3 = PrimitiveStep(primitive_3.metadata.query())
step_4 = PrimitiveStep(primitive_4.metadata.query())

pipeline_input = sample_pipeline.add_input('input dataset')

# For d3m v2018.6.5 version
step_0.add_argument('inputs', ArgumentType.CONTAINER, pipeline_input)
sample_pipeline.add_step(step_0)
denormalize_step_produce = step_0.add_output("produce")

step_1.add_argument('inputs', ArgumentType.CONTAINER, denormalize_step_produce)
sample_pipeline.add_step(step_1)
to_DataFrame_produce = step_1.add_output('produce')


step_4.add_argument('inputs', ArgumentType.CONTAINER, to_DataFrame_produce)
sample_pipeline.add_step(step_4)
extract_produce = step_4.add_output('produce')

'''
step_2.add_argument('inputs', ArgumentType.CONTAINER, to_DataFrame_produce)
sample_pipeline.add_step(step_2)
to_tensor_produce = step_2.add_output('produce')


step_3.add_argument('inputs', ArgumentType.CONTAINER, to_tensor_produce)
sample_pipeline.add_step(step_3)
to_vgg_produce = step_3.add_output('produce')
'''


'''
# For d3m runtime version
sample_pipeline.add_step(step_0)
sample_pipeline.add_step(step_1)
sample_pipeline.add_step(step_2)
sample_pipeline.add_step(step_3)
step_0.add_argument('inputs', ArgumentType.CONTAINER, pipeline_input)
denormalize_step_produce = step_0.add_output('produce')
step_1.add_argument('inputs', ArgumentType.CONTAINER, denormalize_step_produce)
to_DataFrame_produce = step_1.add_output('produce')
step_2.add_argument('inputs', ArgumentType.CONTAINER, to_DataFrame_produce)
to_tensor_produce = step_2.add_output('produce')
step_3.add_argument('inputs', ArgumentType.CONTAINER, to_tensor_produce)
to_vgg_produce = step_3.add_output('produce')
'''
# define the location of the description file of the dataset which should be a json
#dataset_file_path = '/Users/minazuki/Desktop/studies/master/2018Summer/DSBOX/data/datasets/seed_datasets_current/22_handgeometry/22_handgeometry_dataset/datasetDoc.json'
#dataset_file_path =  '/Users/minazuki/Desktop/studies/master/2018Summer/data/66_chlorineConcentration/TRAIN/dataset_TRAIN/datasetDoc.json'
dataset_file_path = '/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/38_sick_dataset/datasetDoc.json'

# step 0: load the dataset description file
dataset = D3MDatasetLoader()
dataset = dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=os.path.abspath(dataset_file_path)))

# step 1: load the dataset with denormalize primitive
result1 = primitive_0.produce(inputs = dataset)
# step 2: transform the dataset to dataframe
result2 = primitive_1.produce(inputs = result1.value) # this should be the input to this primitive
# step 3: transform the dataframe to the ndarry
result3 = primitive_4.produce(inputs = result2.value)

#result4 = d.produce(inputs = result3.value)

print("result3 value is:")
print(result3.value)

# a method to get the metadata (hyper paramters) of a output
#print(result2.value.metadata.query((mbase.ALL_ELEMENTS,0)))