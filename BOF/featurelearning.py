""" DictionaryLearning

Implements the framework for using spherical kmeans to compute large scale
dictionaries, as given in:

@incollection{coates2012learning,
  title={Learning feature representations with k-means},
  author={Coates, Adam and Ng, Andrew Y},
  booktitle={Neural Networks: Tricks of the Trade},
  pages={561--580},
  year={2012},
  publisher={Springer}
}

Code is constructed to minimise memory use, may not always be optimal for small
datasets.
"""

from __future__ import division, print_function
from builtins import range
import numpy as np
from sklearn.base import BaseEstimator


def simple_cov(X):
    """Returns the covariance matrix of the data in X.

    Used instead of numpy.cov when it is necessary to avoid copying a large 
    array. Not recommended for general use, as it operates in place on the input
    array.

    Parameters:
        X : ndarray 
            The data to find the covariance of. Each row is an observation, each 
            column is a feature. This is transposed from the numpy convention, 
            but consistent with scikit learn.

    """
    examples, features = X.shape
    output = np.empty((features, features), dtype='float')
    sample_mean = X.mean(axis=0, keepdims=True)
    X -= sample_mean # Operates in place on original data
    output = np.dot(X.T, X)
    X += sample_mean # Restore the sample mean
    return output/(examples - 1) # 'Unbiased' estimate of covariance.


def normalise_inplace(X, norm_reg=0, brightness=True, avoid_copy=False):
    """Normalise the rows of an array inplace to have zero mean and unit length.

    WARNING: Operates in place, will overwrite input data.

    Parameters:

        X : ndarray 
            Data to normalise in place.

        norm_reg : float, default: 0
            Amount to regularise the brightness normalisation and avoid 
            numerical instability.

        brightness : bool, default: True
            If True normalise the brightness by dividing each row by it's own
            magnitude.

        avoid_copy : bool, default: False
            If True, operate row-by-row and avoid making a copy. Slow.

    Returns:
        None. 

    """

    if avoid_copy:
        for row in X:
            if brightness:
                row -= np.mean(row)
            row /= np.linalg.norm(row) + norm_reg
    else:
        if brightness:
            X -= np.mean(X, axis=1)[:, None]
        X /= (np.linalg.norm(X, axis=1) + norm_reg)[:, None]


class Whiten(BaseEstimator):
    """Whitening transformer that learns a ZCA transform of the provided data.

    Parameters:

        energy : float, default: 0.95
            Amount of energy to retain in the ZCA transform.

        whiten_reg : float, default: 0.1
            Amount to regularise the variance normalisation. Needed to avoid
            numerical instability.

        k : int, default: None
            Number of axes to retain in the ZCA transform. energy is ignored if 
            this is specified.

    """

    def __init__(self, energy=0.95, whiten_reg=0.1, k=None):
        self.energy = energy
        self.whiten_reg = whiten_reg
        self.k = k

    def fit(self, X, y=None):
        """Learn the whitening transform for the given data.

        Parameters:

            X : ndarray 
                Example data to learn the whitening transform. Organised as 1 
                example per row. 

        """
        covariance = simple_cov(X)
        [D, V] = np.linalg.eigh(covariance)

        # Sort eigenvalues from largest to smallest as the ordering is not
        # guaranteed.
        sort_order = D.argsort()[::-1]
        D = D[sort_order]
        V = V[:, sort_order]
        self.D = D
        self.V = V

       
        if self.energy >= 1 or self.energy is None:
            k = X.shape[1]
        if self.k is not None:
            k = self.k
        else:
            # argmax selects the first example when maximum is repeated
            k = np.argmax((D.cumsum()/D.sum()) >= self.energy) + 1
        
        # Discard low energy terms
        D = D[:k]
        V = V[:,:k]

        self.k = k

        self.whiten = (V.dot(np.diag((D + self.whiten_reg)**(-0.5)))).dot(V.T).T

    def transform(self, X, y=None, inplace=False):
        """Return the whitened version of the input data.

        Parameters:

            X : ndarray
                Data to whiten, organised as one example per colummn

            inplace : bool, default: False
                If True, operate inplace on X and avoid copying. Returns None.

        """
        if inplace:
            for i in range(X.shape[0]):
                X[i,:] = X[i,:].dot(self.whiten)
        else:
            return np.dot(X, self.whiten)


def _iterate_spherical(X, centroids, sort_mag=False):
    """Performs a single iteration of spherical kmeans."""
    cluster_assignments = np.dot(X, centroids.T).argmax(axis=1)
    new_centroids = np.zeros_like(centroids)
    
    for i in range(centroids.shape[0]):
        this_cluster = cluster_assignments == i
        if np.sum(this_cluster) == 0: # Reassign zombie cluster
            new_centroids[i,:] = _init_random_selection(X, 1)
        else:
            new_centroids[i,:] = np.sum(X[this_cluster, :], axis=0)

    magnitude = np.linalg.norm(new_centroids, axis=1)
    magnitude[magnitude==0] = 1
    new_centroids /= magnitude[:, None]

    if sort_mag:
        new_centroids = new_centroids[magnitude.argsort()]

    return new_centroids


def _init_random_selection(X, n_clusters):
    """Choose randomly from input patches as initial centroids.
    
    If there are no input patches, return a set of zero centroids. This 
    excludes this group from further consideration."""
    if X.shape[0] < n_clusters:
        return np.zeros((n_clusters, X.shape[1]))
    indices = np.random.randint(X.shape[0], size=n_clusters)
    centroids = X[indices, :].copy()
    return centroids


def _spherical_kmeans(X, n_clusters, iterations):
    """Return centroids after multiple iterations of spherical kmeans."""
    centroids = _init_random_selection(X, n_clusters)
    for i in range(iterations):
        centroids = _iterate_spherical(X, centroids, sort_mag=True)
    return centroids


def _hier_kmeans(X, n_clusters, iterations, levels):
    """Return nested list of centroids corresponding to hierarchical kmeans clustering"""
    centroids = _spherical_kmeans(X, n_clusters, iterations)
    if levels == 1:
        return centroids
    else:
        cluster_assignments = np.dot(X, centroids.T).argmax(axis=1)
        selections = (cluster_assignments == i for i in range(n_clusters))
        lower_levels = [_hier_kmeans(X[select], n_clusters, iterations, levels-1) 
                        for select in selections]
        return [centroids] + lower_levels


def _hier_encode(X, centroids, levels):
    """Return hierarchical encoding of data given a set of centroids"""
    if levels == 1:
        output = np.dot(X, centroids.T).argmax(axis=1)[:, None]
    else:
        output = np.zeros((X.shape[0], levels), dtype='int')
        output[:, 0] = np.dot(X, centroids[0].T).argmax(axis=1)
        for i in range(centroids[0].shape[0]):
            this_cluster = output[:, 0] == i
            output[this_cluster, 1:] = _hier_encode(X[this_cluster, :], 
                                                    centroids[i+1], 
                                                    levels-1)
    return output


class SphericalKMeans(BaseEstimator):
    """Spherical KMeans clustering.

    Parameters:

        n_clusters : int, default: 10
            Number of centroids to learn.

        iterations : int, default: 10
            Number of iterations to use in learning.
    """
    def __init__(self, n_clusters=10, iterations=10):
        self.n_clusters = n_clusters
        self.iterations = iterations

    def fit(self, X, y=None):
        """Cluster the given data."""
        self.centroids = _spherical_kmeans(X, self.n_clusters, self.iterations)

    def transform(self, X, y=None):
        """Return the distance from the input data to each cluster centroid."""
        return np.dot(X, self.centroids.T)

    def predict(self, X, y=None):
        """Return the nearest cluster centroid to each input data point."""
        similarities = np.dot(X, self.centroids.T)
        return similarities.argmax(axis=1)


class HierSKMeans(BaseEstimator):
    """Hierarchical spherical kmeans clustering. 

    Parameters:

        n_clusters : int, default: 10
            Number of centroids to learn.

        iterations : int, default: 10
            Number of iterations to use in learning.

        levels : int, default: 2
            Number of hierarchical levels to use in the encoding.
    """
    def __init__(self, n_clusters=10, iterations=10, levels=2):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.levels = levels

    def fit(self, X, y=None):
        """Learn the hierarchical cluster centroids."""
        self.centroids = _hier_kmeans(X, self.n_clusters, self.iterations, self.levels)

    def transform(self, X, y=None):
        """Return the leaf level codes for each data point."""
        encoding_tree = _hier_encode(X, self.centroids, levels=self.levels)
        return encoding_tree

    def predict(self, X, y=None):
        """Return the index of the nearest centroid for each data point."""
        encoding_tree = _hier_encode(X, self.centroids, levels=self.levels)
        for i in range(1, self.levels):
            encoding_tree[:, -(i+1)] *= self.n_clusters**i  
        return np.sum(encoding_tree, axis=1)









