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


class Augment():

    def __init__(self, approach='rotate'):
        self.approach = approach

    def transform(self, image):
        """Create copies of original image with different rotations"""
        if self.approach == 'rotate':
            return [np.rot90(image, k=i) for i in range(0,4)]
        elif self.approach == 'reflect':
            return [image, np.fliplr(image), np.flipud(image)]
        elif self.approach == 'both':
            images = [np.rot90(image, k=i) for i in range(0,4)]
            images_left = [np.fliplr(im) for im in images]
            images_up = [np.flipud(im) for im in images]
            return images + images_left + images_up
        else:
            return [image]

    def inverse_transform(self, images):
        """Apply the appropriate inverse transforms"""
        if self.approach == 'rotate':
            return [np.rot90(images[i], k=-i) for i in range(0,4)]
        elif self.approach == 'reflect':
            return [images[0], np.fliplr(image[1]), np.flipud(image[2])]
        elif self.approach == 'both':
            base = [np.rot90(images[i], k=-i) for i in range(0,4)]
            left = [np.fliplr(im) for im in images[4:8]]
            left = [np.rot90(left[i], k=-i) for i in range(0,4)]
            up = [np.flipud(im) for im in images[8:]]
            up = [np.rot90(up[i], k=-i) for i in range(0,4)]
            return base + left + up
        else:
            return images

    def __call__(self, image):
        return self.transform(image)


def make_histogram(encoding, words):
    return np.histogram(encoding, bins=words, range=(0, words))[0]


class BagOfFeaturesEncoder(BaseEstimator):
    """ """
    def __init__(self, pixels=7, n_words=10, n_patches=10, energy=0.95,
                 whiten_reg=0.1, norm_reg=0.01, iterations=10, levels=1,
                 verbose=False, augment=None):
        self.pixels = pixels
        self.n_words = n_words
        self.n_patches = n_patches
        self.energy = energy
        self.whiten_reg = whiten_reg
        self.iterations = iterations
        self.norm_reg = norm_reg
        self.levels = levels
        self.verbose = verbose
        self.augment = augment

    def fit(self, images, n_images=None, y=None): # Add option to incorporate prebuilt whiten etc operators.
        """
        images: an iterable of images as 2d arrays """
        self.augment_ = Augment(self.augment)
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

    def predict(self, images, y=None, pool=False):
        """Compute the histogram of visual word counts."""
        histograms = []
        i = 0
        total_words = self.n_words**self.levels
        for image in images:
            i += 1
            if self.verbose:
                print(i)
            augmented = self.augment_(image)
            encoded = [self.transform(im) for im in augmented]
            histogram = [make_histogram(code, total_words) for code in encoded]
            if pool:
                histograms.append(sum(histogram))
            else:
                histograms.extend(histogram)
        return np.vstack(histograms)

    def predict_pixels(self, image, y=None):
        """Compute the augmented one hot encoding for the input image."""
        predicted = []
        i = 0
        total_words = self.n_words**self.levels
        augmented = self.augment_(image)
        encoded = [self.transform(im, reshape=True) for im in augmented]
        encoded = sum(self.augment_.inverse_transform(encoded))
        return encoded

    def transform(self, image, y=None, reshape=False):
        """Transform a single image to per pixel encoding.

        reshape: reshape output from 2d array of distances for each patch to a
            3d array of rows x columns x feature responses. """
        
        patches = self.preprocess(image)
        encoded = self.cluster.predict(patches)

        # reshape to original image size, minus pixels lost at the boundary
        if reshape:
            rows, cols = np.array(image.shape[:2]) - self.pixels + 1
            # Create a 1-hot encoding (sparse binary array)
            output = np.zeros((encoded.shape[0], self.n_words**self.levels), 
                              dtype='bool')
            output[np.arange(encoded.shape[0]), encoded] = True
            encoded = output.reshape((rows, cols, -1))

        return encoded




