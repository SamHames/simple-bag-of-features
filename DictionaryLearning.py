"""


"""
from __future__ import division, print_function
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.base import TransformerMixin, BaseEstimator


class Whiten(BaseEstimator):
    """ Whitens the given data. Assumes input patches are already normalised
    to zero mean at least zero mean"""
    def __init__(self, energy=0.95, whiten_reg=0.1, k=None, method='ZCA'):
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
        indices = np.random.randint(data.shape[0], size=n_centroids)
        centroids = X[indices, :]

        centroid_update = np.zeros(centroids.shape)

        for i in range(max_iter):
            proj = np.dot(X, centroids.T)
            proj_max_loc = proj.argmax(axis=1)
            proj_max = proj.max(axis=1)
            del proj
            # This is the weighted update given in Coates + Ng (2012)
            s = np.zeros((n_centroids, X.shape[0]))
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


class BagOfFeaturesEncoder(BaseEstimator):
    """ """
    def __init__(self, pixels=7, n_words=10, n_patches=10, energy=0.95,
                 whiten_reg=0.1, variance_reg=1, max_iter=10):
        self.pixels = pixels
        self.n_words = n_words
        self.n_patches = n_patches
        self.energy = energy
        self.whiten_reg = whiten_reg
        self.max_iter = max_iter
        self.variance_reg = variance_reg

    def fit(self, images, n_images=None, y=None): # Add option to incorporate prebuilt whiten etc operators.
        """
        images: an iterable of images as 2d arrays """
        # Extract patches from images
        patches = collect_normalised_patches(images, n_images=n_images,
                                             variance_reg=self.variance_reg)
        self.whiten = Whiten(energy=self.energy, whiten_reg=self.whiten_reg)
        self.whiten.fit(patches)
        self.whiten.transform(patches, inplace=True)
        self.cluster = SphericalKMeans(n_clusters=n_words, max=self.max_iter)

    def predict(self, images, y=None): # Could take an extra argument for different encoding strategies here....
        """ """
        histograms = []
        for image in images:
            patches = extract_patches_2d(image, 
                                         patch_size=(self.pixels, self.pixels))
            patches = patches.reshape((-1, self.pixels**2))
            normalise_inplace(patches, self.whiten_reg)
            patches = self.whiten.transform(patches)
            codes = self.whiten.predict(patches)
            [histogram, bins] = np.histogram(codes,
                                             bins=self.n_words,
                                             range=(0, self.n_words))
            histograms.append(histogram.astype('float')/histogram.sum())
        return np.vstack(histograms)


def collect_normalised_patches(images, n_images, pixels=7,
                               n_patches=100, variance_reg=1):
    """Collect fixed number of patches from random locations in each image.

    Parameters:
        images - iterable of images as 2d numpy floating point numpy arrays
        n_images - number of images. Needed to preallocate output array.
        stacks - list of stacks as w x h x depth pixels
        pixels - side of patch in pixels (square patches assumed)
        max_patches - the number of patches to harvest from each image

    Returns:
        patch_array: array of n patches x m features (unwraps each patch to a 1d vector)

    Uses scikit learn patch extractor for each layer of each stack passed in.
    Patches with zero variance are discarded and replaced.

    Can specify the patch size and number of patches per image as per sklearn extract
    patches.

    """
    # Preallocate output array for memory efficiency
    patches = np.zeros((n_images*max_patches,patch_size**2), dtype='float')
    patch_index = 0
    for image in images:
        these_patches = extract_patches_2d(image,
                                           patch_size=(pixels, pixels),
                                           patches=n_patches)
        these_patches = these_patches.reshape((n_patches, pixels**2))
        normalise_inplace(these_patches, 
                          variance_reg=variance_reg, 
                          brightness=True)
        patches[patch_index:patch_index + n_patches] = these_patches
    return patches

def simple_cov(x):
    """Computes the covariance matrix of the data in X.

    Avoids the temporary array created when using numpy.cov, and exploits the
    fact that the data here is already centered around zero.

    Assumes data is real, and each observation is centered around zero.

    Also, deals with data arranged as rows of independant observations, unlike 
    numpy.cov
    """
    rows = x.shape
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






