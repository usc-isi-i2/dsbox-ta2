from d3m.container.dataset import D3MDatasetLoader, Dataset, CSVLoader
from common_primitives.denormalize import DenormalizePrimitive, Hyperparams as hyper_DE
import os
h1 = hyper_DE.defaults()
primitive_0 = DenormalizePrimitive(hyperparams=h1)

dataset_file_path = '/nfs1/dsbox-repo/data/datasets-v31/seed_datasets_current' \
                    '/22_handgeometry/22_handgeometry_dataset/datasetDoc.json'
#dataset_file_path = '/Users/minazuki/Desktop/studies/master/2018Summer/data/38_sick_new/38_sick_dataset/datasetDoc.json'
# step 0: load the dataset description file
dataset = D3MDatasetLoader()
dataset = dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=os.path.abspath(dataset_file_path)))

# step 1: load the dataset with denormalize primitive
result1 = primitive_0.produce(inputs=dataset)
