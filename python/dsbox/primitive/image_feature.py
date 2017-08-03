""" Feature generation for images
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from dsbox.planner.levelone import Primitive

import numpy as np

class PcaImageFeature(BaseEstimator, TransformerMixin):
    """
    Image Feature Generation using PCA

    Parameters
    ----------
    explained_variance : float, default: 0.9, range: 0 < explained_variance < 1
        Controls the number of features to generate.
        Higher explained_variance implies more features.
    """
    def __init__(self, explained_variance=0.9):
        self.explained_variance = explained_variance
        self.pca = None
        self._annotation = None

    def fit(self, image_tensor, y=None):
        """Fit the model with image_tensor.
        image_tensor : array-like, shape (n_samples, *)
        """
        data = image_tensor.reshape(image_tensor.shape[0], -1)
        if self.explained_variance < 0 or self.explained_variance >= 1:
            self.explained_variance = 0.9
        self.pca = PCA(n_components=self.explained_variance, svd_solver='full')
        self.pca.fit(data)

    def transform(self, image_tensor):
        """Apply dimensionality reduction to X.
        """
        data = image_tensor.reshape(image_tensor.shape[0], -1)
        return self.pca.transform(data)

    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'PcaImageFeature'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = ''
        self._annotation.ml_algorithm = ['Dimension Reduction']
        self._annotation.tags = ['feature_extraction' , 'image']
        return self._annotation


class ColorRegionImageFeature(BaseEstimator, TransformerMixin):
    """
    Image Feature Generation based on color cluster sizes

    Parameters
    ----------
    n_cluster : int, default: 8, range: n_cluster > 0
        Number of color region clusters.
    """
    def __init__(self, n_cluster=8):
        self.n_cluster = n_cluster
        self.kmeans = None
        self._annotation = None

    def fit(self, image_tensor, y=None):
        """Fit the model with image_tensor"""
        pass

    def transform(self, image_tensor, y=None):
        """Apply color cluster based dimension reduction to image_tensor"""
        if len(image_tensor) == 3:
            # Assume gray scale image shape = (n_samples, row, col)
            pixel_data = image_tensor.reshape(image_tensor.shape[0], -1)
        elif len(image_tensor) == 4:
            # Assume color scale image shape = (n_samples, row, col, color)
            pixel_data = image_tensor.reshape(image_tensor.shape[0], -1, image_tensor.shape[3])
        else:
            raise Exception('ColorRegionImageFeature: not able to process shape: {}'.
                            format(image_tensor.shape))

        self.kmeans = MiniBatchKMeans()
        results = []
        for i in range(pixel_data.shape[0]):
            self.kmeans.fit(pixel_data[i, :, :])
            centers = np.uint8(self.kmeans.cluster_centers_)
            unique_values, unique_counts = np.unique(self.kmeans.labels_, return_counts=True)
            sort_index = np.argsort(unique_counts)
            # Reverse order. Biggest cluster goes first
            sort_index = sort_index[::-1]
            result = np.zeros(4 * len(unique_values))
            for j in range(len(sort_index)):
                result[4 * j] = unique_counts[sort_index[j]]
                color = centers[unique_values[sort_index[j]], :]
                result[4 * j + 1:4 * j + 4] = color
            results.append(result)
        return np.vstack(results)

    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'ColorRegionImageFeature'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = ''
        self._annotation.ml_algorithm = ['Clustering']
        self._annotation.tags = ['feature_extraction' , 'image']
        return self._annotation
