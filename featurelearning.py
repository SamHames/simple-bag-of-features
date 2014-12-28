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
    """Computes the covariance matrix of the data in X.

    Avoids the temporary array created when using numpy.cov.

    Assumes data is real, and each observation is centered around zero.

    Also, deals with data arranged as rows of independant observations, unlike
    numpy.cov defaults.

    Will be significantly slower than np.cov(). Necessary to avoid temporary 
    matrices when dealing with arrays near the size of ram.

    Not recommended unless input values are bounded and near the sample average 
    . Adds and subtracts this in place and possible numerical issues.
    """
    examples, features = X.shape
    output = np.empty((features, features), dtype='float')
    sample_mean = X.mean(axis=0, keepdims=True)
    X -= sample_mean # Operates in place on original data
    output = np.dot(X.T, X)
    X += sample_mean # Restore the sample mean
    return output/(examples - 1) # 'Unbiased' estimate of covariance.


def normalise_inplace(X, norm_reg=0, brightness=True, avoid_copy=False):
    """Normalise the rows of an array in place to have zero mean and unit length.

    avoid_copy: operate row by row instead of using vectorised numpy expression.
        Slower, but useful to avoid allocating a very large temporary matrix.
        When using spherical kmeans it is possible to work with 10's of millions
        of examples, an unexpected temporary array can easily exhaust available
        ram.

    Notes
    -----
    WARNING: Operates in place, will overwrite input data.
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
    """ Whitens the given data using ZCA transform"""
    def __init__(self, energy=0.95, whiten_reg=0.1, k=None):
        """
        energy: Since rows are assumed zero centered, low energy eigenvalues are
            expected, so energy should be less than 1.0
        whiten_reg:
        k: Number of eigenvectors to retain in the transform. If specified, the
        energy term is ignored."""
        self.energy = energy
        self.whiten_reg = whiten_reg
        self.k = k

    def fit(self, X, y=None):
        """Learn the whitening transform matrix by PCA."""
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
        """
        inplace: Update the rows in place, one at a time. Slower, but avoids
            making a temporary copy of a large array. """
        if inplace:
            for i in range(X.shape[0]):
                X[i,:] = X[i,:].dot(self.whiten)
        else:
            return np.dot(X, self.whiten)


def _iterate_spherical(X, centroids, sort_mag=False):
    """Performs a single step of spherical kmeans."""
    similarities = np.dot(X, centroids.T)
    cluster_assignments = np.argmax(similarities, axis=1)
    new_centroids = np.zeros_like(centroids)
    
    for i in range(centroids.shape[0]):
        this_cluster = cluster_assignments == i
        if np.sum(this_cluster) == 0: # Reassign zombie cluster
            new_centroids[i,:] = _init_random_selection(X)
        else:
            new_centroids[i,:] = np.sum(X[this_cluster, :], axis=0)

    magnitude = np.linalg.norm(new_centroids, axis=1)
    magnitude[magnitude==0] = 1
    new_centroids /= magnitude[:, None]

    if sort_mag:
        new_centroids = new_centroids[magnitude.argsort()]

    return new_centroids


def _init_random_selection(X, n_clusters=None):
    """Choose randomly from input patches as initial centroids."""
    indices = np.random.randint(X.shape[0], size=n_clusters)
    centroids = X[indices, :].copy()
    return centroids


def _spherical_kmeans(X, n_clusters, iterations):
    centroids = _init_random_selection(X, n_clusters=n_clusters)
    for i in range(iterations):
        centroids = _iterate_spherical(X, centroids, sort_mag=True)
    return centroids


def _hier_kmeans(X, n_clusters, iterations, levels):
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
    """Assumes normalised input (zero mean and unit variance or magnitude)"""
    def __init__(self, n_clusters=10, iterations=10):
        self.n_clusters = n_clusters
        self.iterations = iterations

    def fit(self, X, y=None):
        """ """
        self.centroids = _spherical_kmeans(X, self.n_clusters, self.iterations)

    def transform(self, X, y=None):
        """ """
        return np.dot(X, self.centroids.T)

    def predict(self, X, y=None):
        """ """
        similarities = np.dot(X, self.centroids.T)
        return similarities.argmax(axis=1)


class HierSKMeans(BaseEstimator):
    def __init__(self, n_clusters=10, iterations=10, levels=2):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.levels = levels

    def fit(self, X, y=None):
        """ """
        self.centroids = _hier_kmeans(X, self.n_clusters, self.iterations, self.levels)

    def transform(self, X, y=None):
        encoding_tree = _hier_encode(X, self.centroids, levels=self.levels)
        return encoding_tree

    def predict(self, X, y=None):
        encoding_tree = _hier_encode(X, self.centroids, levels=self.levels)
        for i in range(1, self.levels):
            encoding_tree[:, -(i+1)] *= self.n_clusters**i  
        return np.sum(encoding_tree, axis=1)









