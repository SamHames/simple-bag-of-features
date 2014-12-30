""" image_encoding

Applies bag of features encoding to images, by learning transforms from large
numbers of randomly sampled patches.

"""

from __future__ import division, print_function
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.base import BaseEstimator
from featurelearning import Whiten, SphericalKMeans, normalise_inplace, HierSKMeans


def collect_normalised_patches(images, n_images, pixels=7,
                               n_patches=100, norm_reg=1):
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
    patches = np.empty((n_images*n_patches, pixels**2), dtype='float')
    patch_index = 0
    for image in images:
        image = np.asfarray(image)
        these_patches = extract_patches_2d(image,
                                           patch_size=(pixels, pixels),
                                           max_patches=n_patches)
        these_patches = these_patches.reshape((n_patches, pixels**2))
        normalise_inplace(these_patches,
                          norm_reg=norm_reg,
                          brightness=True)
        patches[patch_index:patch_index + n_patches] = these_patches
        patch_index += n_patches
    return patches


class BagOfFeaturesEncoder(BaseEstimator):
    """ """
    def __init__(self, pixels=7, n_words=10, n_patches=10, energy=0.95,
                 whiten_reg=0.1, norm_reg=0.01, iterations=10, levels=1,
                 verbose=False):
        self.pixels = pixels
        self.n_words = n_words
        self.n_patches = n_patches
        self.energy = energy
        self.whiten_reg = whiten_reg
        self.iterations = iterations
        self.norm_reg = norm_reg
        self.levels = levels
        self.verbose = verbose

    def fit(self, images, n_images=None, y=None): # Add option to incorporate prebuilt whiten etc operators.
        """
        images: an iterable of images as 2d arrays """
        self.whiten = Whiten(energy=self.energy, whiten_reg=self.whiten_reg)
        if self.levels > 1:
            self.cluster = HierSKMeans(n_clusters=self.n_words,
                                       iterations=self.iterations,
                                       levels = self.levels)
        else:
            self.cluster = SphericalKMeans(n_clusters=self.n_words,
                                           iterations=self.iterations)

        # Extract patches from images
        patches = collect_normalised_patches(images, n_images=n_images,
                                             pixels=self.pixels,
                                             norm_reg=self.norm_reg,
                                             n_patches=self.n_patches)
        self.whiten.fit(patches)
        self.whiten.transform(patches, inplace=True)
        self.cluster.fit(patches)

    def preprocess(self, image):
        patches = extract_patches_2d(image,
                                         patch_size=(self.pixels, self.pixels))
        patches = patches.reshape((-1, self.pixels**2))
        normalise_inplace(patches, self.whiten_reg)
        return self.whiten.transform(patches)

    def predict(self, images, y=None):
        """Compute the histogram of visual word counts, using one hot encoding."""
        histograms = []
        i = 0
        for image in images:
            i += 1
            if self.verbose:
                print(i)
            patches = self.preprocess(image)
            codes = self.cluster.predict(patches)
            [histogram, bins] = np.histogram(codes,
                                             bins=self.n_words**self.levels,
                                             range=(0, self.n_words**self.levels))
            histograms.append(histogram)
        return np.vstack(histograms)

    def transform(self, images, y=None, reshape=True):
        """Compute the feature similarity for every pixel in the image.

        reshape: reshape output from 2d array of distances for each patch to a
            3d array of rows x columns x feature responses. """

        transformed = []
        for image in images:
            patches = self.preprocess(image)
            similarities = self.cluster.transform(patches)

            # reshape to original image size, minus pixels lost at the boundary
            if reshape and self.levels == 1: # ignore reshape if using hierarchical.
                rows, cols = np.array(image.shape[:2]) - self.pixels + 1
                similarities = similarities.reshape((rows, cols, -1))

            transformed.append(similarities)
        return transformed


class TriangleEncoder(BaseEstimator):
    def __init__(self, alpha=0):
        self.alpha = alpha

    def fit(self):
        pass

    def transfomr(self, X, pool=True):
        output = np.asarray(X)
        output -= self.alpha
        output[output<0] = 0
        if pool: # Concatenate the results into vector
            output = output.sum(axis=0)
            output /= output.sum()
        return output



