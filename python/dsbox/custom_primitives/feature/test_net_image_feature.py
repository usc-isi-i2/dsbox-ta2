""" Sample Usage to demonstrate deep learning based feature extraction.
"""
from keras.preprocessing import image
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from dsbox.custom_primitives.feature import ResNet50ImageFeature, Vgg16ImageFeature
import csv
import numpy as np
import os

# data_dir = '/home/ktyao/Projects/DSBox/dev/dsbox-data/r_22/data'
data_dir = None

if data_dir is None:
    print('To run this script set "data_dir" to point to the D3M r_22/data directory.')
    import sys
    sys.exit()


image_dir = os.path.join(data_dir, 'raw_data')


def load_images(csvFile, image_size=(224, 224)):
    small_images = []
    with open(os.path.join(data_dir, csvFile), 'r') as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        next(reader)
        for row in reader:
            print(row[1])
            filepath = os.path.join(image_dir, row[1])
            small = image.load_img(filepath, target_size=image_size)
            small_images.append(small)
    # data = np.vstack([x.reshape(-1) for x in small_images])
    # return data
    return small_images

def as_tensor(image_list):
    shape = (len(image_list), ) + image.img_to_array(image_list[0]).shape
    result = np.empty(shape)
    for i in range(len(image_list)):
        result[i] = image.img_to_array(image_list[i])
    return result

def load_targets(csvFile):
    targets = []
    with open(os.path.join(data_dir, csvFile), 'r') as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        next(reader)
        for row in reader:
            targets.append(float(row[1]))
    targets = np.asarray(targets)
    return targets

image_list = load_images('trainData.csv')
all_images = as_tensor(image_list)

training_images = all_images[:150]
testing_images = all_images[150:]

all_targets = load_targets('trainTargets.csv')
training_targets = all_targets[:150]
testing_targets = all_targets[150:]

print()
for index in range(5):
    print('VGG16 index={}'.format(index))
    extractor = Vgg16ImageFeature(index)
    print('layer num = {}'.format(extractor.layer_number))
    extractor.fit(training_images)
    training_encoded_data = extractor.transform(training_images)
    testing_encoded_data = extractor.transform(testing_images)
    print('num features = {}'.format(training_encoded_data.shape[1]))

    tree = DecisionTreeRegressor()
    tree.fit(training_encoded_data, training_targets)
    print('    VGG16 Tree R2 training score = {}'
          .format(tree.score(training_encoded_data, training_targets)))
    print('    VGG16 Tree R2 testing  score = {}'
          .format(tree.score(testing_encoded_data, testing_targets)))

    tree = DecisionTreeRegressor(max_depth=4)
    tree.fit(training_encoded_data, training_targets)
    print('    VGG16 Tree MaxLevel=4 R2 training score = {}'
          .format(tree.score(training_encoded_data, training_targets)))
    print('    VGG16 Tree MaxLevel=4 R2 testing  score = {}'
          .format(tree.score(testing_encoded_data, testing_targets)))

    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(training_encoded_data, training_targets)
    print('    VGG16 Tree MaxLevel=3 R2 training score = {}'
          .format(tree.score(training_encoded_data, training_targets)))
    print('    VGG16 Tree MaxLevel=3 R2 testing  score = {}'
          .format(tree.score(testing_encoded_data, testing_targets)))

    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(training_encoded_data, training_targets)
    print('    VGG16 Tree MaxLevel=2 R2 training score = {}'
          .format(tree.score(training_encoded_data, training_targets)))
    print('    VGG16 Tree MaxLevel=2 R2 testing  score = {}'
          .format(tree.score(testing_encoded_data, testing_targets)))

    svm = SVR()
    scaler = preprocessing.MinMaxScaler().fit(training_encoded_data)
    svm.fit(scaler.transform(training_encoded_data), training_targets)
    print('    VGG16 MinMaxScale Svm R2 training score = {}'
          .format(svm.score(scaler.transform(training_encoded_data), training_targets)))
    print('    VGG16 MinMaxScale Svm R2 testing  score = {}'
          .format(svm.score(scaler.transform(testing_encoded_data), testing_targets)))

    svm = SVR()
    svm.fit(training_encoded_data, training_targets)
    print('    VGG16 Svm R2 training score = {}'
          .format(svm.score(training_encoded_data, training_targets)))
    print('    VGG16 Svm R2 testing  score = {}'
          .format(svm.score(testing_encoded_data, testing_targets)))

print()
for index in range(11):
    print('ResNet50 index={}'
          .format(index))
    extractor = ResNet50ImageFeature(index)
    print('layer num = {}'
          .format(extractor.layer_number))
    extractor.fit(training_images)
    training_encoded_data = extractor.transform(training_images)
    testing_encoded_data = extractor.transform(testing_images)
    print('num features = {}'.format(training_encoded_data.shape[1]))

    tree = DecisionTreeRegressor()
    tree.fit(training_encoded_data, training_targets)
    print('    ResNet Tree R2 training score = {}'
          .format(tree.score(training_encoded_data, training_targets)))
    print('    ResNet Tree R2 testing  score = {}'
          .format(tree.score(testing_encoded_data, testing_targets)))

    tree = DecisionTreeRegressor(max_depth=4)
    tree.fit(training_encoded_data, training_targets)
    print('    ResNet Tree MaxLevel=4 R2 training score = {}'
          .format(tree.score(training_encoded_data, training_targets)))
    print('    ResNet Tree MaxLevel=4 R2 testing  score = {}'
          .format(tree.score(testing_encoded_data, testing_targets)))

    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(training_encoded_data, training_targets)
    print('    ResNet Tree MaxLevel=3 R2 training score = {}'
          .format(tree.score(training_encoded_data, training_targets)))
    print('    ResNet Tree MaxLevel=3 R2 testing  score = {}'
          .format(tree.score(testing_encoded_data, testing_targets)))

    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(training_encoded_data, training_targets)
    print('    ResNet Tree MaxLevel=2 R2 training score = {}'
          .format(tree.score(training_encoded_data, training_targets)))
    print('    ResNet Tree MaxLevel=2 R2 testing  score = {}'
          .format(tree.score(testing_encoded_data, testing_targets)))

    svm = SVR()
    scaler = preprocessing.MinMaxScaler().fit(training_encoded_data)
    svm.fit(scaler.transform(training_encoded_data), training_targets)
    print('    ResNet MinMaxScale Svm R2 training score = {}'
          .format(svm.score(scaler.transform(training_encoded_data), training_targets)))
    print('    ResNet MinMaxScale Svm R2 testing  score = {}'
          .format(svm.score(scaler.transform(testing_encoded_data), testing_targets)))

    svm = SVR()
    svm.fit(training_encoded_data, training_targets)
    print('    ResNet Svm R2 training score = {}'
          .format(svm.score(training_encoded_data, training_targets)))
    print('    ResNet Svm R2 testing  score = {}'
          .format(svm.score(testing_encoded_data, testing_targets)))
