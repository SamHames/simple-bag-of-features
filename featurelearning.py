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

"""
from __future__ import division, print_function
import numpy as np
from sklearn.base import BaseEstimator


def simple_cov(x):
    """Computes the covariance matrix of the data in X.

    Avoids the temporary array created when using numpy.cov, and exploits the
    fact that the data here is already centered around zero.

    Assumes data is real, and each observation is centered around zero.

    Also, deals with data arranged as rows of independant observations, unlike
    numpy.cov
    """
    rows = x.shape[0]
    output = np.dot(x.T, x)
    return output/(rows - 1) # 'Unbiased' estimate of covariance.


def normalise_inplace(data, variance_reg=0, brightness=True, avoid_copy=False):
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
        for row in data:
            if brightness:
                row -= np.mean(row)
            row /= np.sqrt(np.var(row, axis=1) + variance_reg)
    else:
        if brightness:
            data -= np.mean(data, axis=1)
        data /= np.sqrt(np.var(data, axis=1) + variance_reg)


class Whiten(BaseEstimator):
    """ Whitens the given data. Assumes input patches are already normalised
    to zero mean at least zero mean"""
    def __init__(self, energy=0.95, whiten_reg=0.1, k=None):
        """
        energy:
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

        # Discard low energy terms
        if self.k is not None:
            k = self.k
        else:
            k = np.argmax((D.cumsum()/D.sum()) > self.energy)

        D = D[:k]
        V = V[:,:k]
        self.whiten = (V.dot(np.diag((D + self.whiten_reg)**(-0.5)))).dot(V.T).T

    def transform(self, X, y=None, inplace=False):
        """
        inplace: Update the rows in place, one at a time. Slower, but avoids
            making a temporary copy of a large array. """
        if inplace:
            for row in X:
                whitened = row.dot(self.whiten)
                row = whitened
        else:
            X = np.dot(X, self.whiten)
            return X


class SphericalKMeans(BaseEstimator):
    """Assumes normalised input (zero mean and unit variance or magnitude)"""
    def __init__(self, n_clusters=10, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """ """
        # Initialise with randomly selected examples
        indices = np.random.randint(X.shape[0], size=self.n_clusters)
        centroids = X[indices, :]

        centroid_update = np.zeros(centroids.shape)

        for i in range(self.max_iter):
            proj = np.dot(X, centroids.T)
            proj_max_loc = proj.argmax(axis=1)
            proj_max = proj.max(axis=1)
            del proj
            # This is the weighted update given in Coates + Ng (2012)
            s = np.zeros((self.n_clusters, X.shape[0]))
            s[proj_max_loc, np.arange(X.shape[0])] = proj_max

            np.dot(s, X, out=centroid_update)
            del s
            centroids += centroid_update
            mag = np.sqrt((centroids**2).sum(axis=1))
            centroids /= mag[:, None]

        self.centroids = centroids[mag.argsort()]

    def transform(self, X, y=None):
        """ """
        return np.dot(X, self.centroids.T)

    def predict(self, X, y=None):
        """ """
        similarities = np.dot(X, self.centroids.T)
        return similarities.argmax(axis=1)









