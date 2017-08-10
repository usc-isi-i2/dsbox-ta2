""" Feature generation based on deep learning for images
"""

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.misc import imresize

from keras.models import Model
import keras.applications.resnet50 as resnet50
import keras.applications.vgg16 as vgg16

from dsbox.planner.levelone import Primitive

import numpy as np

class ResNet50ImageFeature(BaseEstimator, TransformerMixin):
    """
    Image Feature Generation using pretrained deep neural network RestNet50.

    Parameters
    ----------
    layer_index : int, default: 0, domain: range(11)
        Layer of the network to use to generate features. Smaller
        indices are closer to the output layers of the network.

    resize_data : Boolean, default: True, domain: {True, False}
        If True resize images to 224 by 224.
    """

    RESNET50_MODEL = None

    def __init__(self, layer_index=0, preprocess_data=True, resize_data=False):
        if ResNet50ImageFeature.RESNET50_MODEL is None:
            ResNet50ImageFeature.RESNET50_MODEL = resnet50.ResNet50(weights='imagenet')
        self.layer_numbers = [-2, -4, -8, -11, -14, -18, -21, -24, -30, -33, -36]
        if layer_index < 0:
            layer_index = 0
        elif layer_index > len(self.layer_numbers):
            self.layer_numbers = len(self.layer_numbers)-1

        self.layer_index = layer_index
        self.layer_number = self.layer_numbers[self.layer_index]
        self.preprocess_data = preprocess_data

        self.resize_data = resize_data

        self.org_model = ResNet50ImageFeature.RESNET50_MODEL
        self.model = Model(self.org_model.input,
                           self.org_model.layers[self.layer_number].output)
        self._annotation = None

    def fit(self, X=None, y=None):
        """Fit the model with X"""
        pass

    def preprocess(self, image_tensor):
        """Preprocess image data by modifying it directly"""
        resnet50.preprocess_input(image_tensor)

    def transform(self, image_tensor, y=None):
        """Apply neural network-based feature extraction to image_tensor"""
        # preprocess() modifies the data. For now just copy the data.
        if not len(image_tensor.shape) == 4:
            raise ValueError('Expect shape to have 4 dimension')

        resized = False
        if self.resize_data:
            if not (image_tensor.shape[1] == 244 and image_tensor.shape[2] == 244):
                resized = True
                y = np.empty((image_tensor.shape[0], 224, 224, 3))
                for index in range(image_tensor.shape[0]):
                    y[index] = imresize(image_tensor[index], (224, 224))
                image_tensor = y

        # preprocess() modifies the data. For now just copy the data.
        if self.preprocess_data:
            if resized:
                # Okay to modify image_tensor, since its not input
                data = image_tensor
            else:
                data = image_tensor.copy()
            self.preprocess(data)
        else:
            data = image_tensor
        result = self.model.predict(data)
        return result.reshape(result.shape[0], -1)
    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'ResNet50ImageFeature'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = ''
        self._annotation.ml_algorithm = ['Deep Learning']
        self._annotation.tags = ['feature_extraction' , 'image']
        return self._annotation


class Vgg16ImageFeature(BaseEstimator, TransformerMixin):
    """
    Image Feature Generation using pretrained deep neural network VGG16.

    Parameters
    ----------
    layer_index : int, default: 0, domain: range(5)
        Layer of the network to use to generate features. Smaller
        indices are closer to the output layers of the network.

    resize_data : Boolean, default: True, domain: {True, False}
        If True resize images to 224 by 224.
    """

    VGG16_MODEL = None

    def __init__(self, layer_index=0, preprocess_data=True, resize_data=False):
        if Vgg16ImageFeature.VGG16_MODEL is None:
            Vgg16ImageFeature.VGG16_MODEL = vgg16.VGG16(weights='imagenet', include_top=False)
        self.layer_numbers = [-1, -5, -9, -13, -16]
        if layer_index < 0:
            layer_index = 0
        elif layer_index > len(self.layer_numbers):
            self.layer_index = len(self.layer_numbers)-1

        self.layer_index = layer_index
        self.layer_number = self.layer_numbers[self.layer_index]
        self.preprocess_data = preprocess_data

        self.resize_data = resize_data

        self.org_model = self.VGG16_MODEL
        self.model = Model(self.org_model.input,
                           self.org_model.layers[self.layer_number].output)
        self._annotation = None

    def fit(self, X=None, y=None):
        """Fit the model with X"""
        pass

    def preprocess(self, image_tensor):
        """Preprocess image data by modifying it directly"""
        vgg16.preprocess_input(image_tensor)

    def transform(self, image_tensor, y=None):
        """Apply neural network-based feature extraction to image_tensor"""
        if not len(image_tensor.shape) == 4:
            raise ValueError('Expect shape to have 4 dimension')

        resized = False
        if self.resize_data:
            if not (image_tensor.shape[1] == 244 and image_tensor.shape[2] == 244):
                resized = True
                y = np.empty((image_tensor.shape[0], 224, 224, 3))
                for index in range(image_tensor.shape[0]):
                    y[index] = imresize(image_tensor[index], (224, 224))
                image_tensor = y

        # preprocess() modifies the data. For now just copy the data.
        if self.preprocess_data:
            if resized:
                # Okay to modify image_tensor, since its not input
                data = image_tensor
            else:
                data = image_tensor.copy()
            self.preprocess(data)
        else:
            data = image_tensor
        result = self.model.predict(data)
        return result.reshape(result.shape[0], -1)

    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'Vgg16ImageFeature'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = ''
        self._annotation.ml_algorithm = ['Deep Learning']
        self._annotation.tags = ['feature_extraction' , 'image']
        return self._annotation
