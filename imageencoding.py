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
                               n_patches=100, norm_reg=1.0):
    """Collect fixed number of patches from random locations in each image.

    Parameters:

        images : iterable of ndarrays
            Iterable of images as 2d numpy floating point numpy arrays.

        n_images : int 
            Total number of images. Needed to preallocate output array.

        pixels : int, default: 7
            Side of patch in pixels. Only square patches are supported.

        max_patches : int, default: 100
            The number of patches to harvest from each image

        norm_reg : float, default 1.0
            The amount to regularise the brightness normalisation.

    Returns:

        patch_array : ndarray, size (n_images*n_patches, pixels**2)

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
    """Apply simple image transforms to generate additional image examples.

    Parameters:

        approach : str, default: 'rotate'
            What style of augmentation to apply. If None, use the original image
            only. If 'rotate', use the original image and the 90, 180 and 270deg
            rotations. If 'mirror', use horizontal and vertical reflections. If
            'both', apply rotations and then horizontal and vertical reflections
            to generate 12 images. 

    """

    def __init__(self, approach='rotate'):
        self.approach = approach

    def transform(self, image):
        """Return a list of the augmented variants of the input image."""
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
        """Return a list of images with the inverse transform applied."""
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
    """Return a histogram of word counts from the input encoding."""
    return np.histogram(encoding, bins=words, range=(0, words))[0]


def _combine_proj_whiten(hier_centroids, whiten_matrix, levels):
    """Premultiplies the dictionary by the whitening matrix.
    """
    if levels == 1:
        return np.dot(whiten_matrix, hier_centroids.T).T
    else:
        for i in range(hier_centroids[0].shape[0]):
            head = np.dot(whiten_matrix, hier_centroids[0].T).T
            rest = [_combine_proj_whiten(centroids, whiten_matrix, 
                                         levels-1) for centroids in hier_centroids[1:]]
            return [head] + rest


class BagOfFeaturesEncoder(BaseEstimator):
    """Transformer for a bag of features encoding of an image.

    Implements the pipeline suggested in "Learning feature representations with
    k-means", Coates and Ng, 2012 in Neural Networks: Tricks of the Trade.

    Parameters:

        pixels : int, default: 7
            Size of patches to extract.

        n_words : int, default: 10
            Number of visual words to learn at each layer of the hierarchical
            encoding.

        n_patches : int, default: 10
            The number of patches to extract from each image when learning the
            whitening transform and cluster centroids.

        energy : float, default: 0.95
            Energy to retain in the ZCA whitening transform.

        whiten_reg : float, default: 0.1
            Regularisation of the variance normalisation in whitening.

        norm_reg : float, default: 1
            Regularisation factor for brightness normalisation.

        iterations : int, default: 10
            Number of iterations to use in learning the centroids.

        levels : int, default: 1
            Number of hierarchical levels to use for encoding.

        verbose : bool, default: False
            If True, print progress numbers when calling predict method.

        augment : str, default: None
            Style of image augmentation to use.
    """
    def __init__(self, pixels=7, n_words=10, n_patches=10, energy=0.95,
                 whiten_reg=0.1, norm_reg=1, iterations=10, levels=1,
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

    def fit(self, images, n_images, y=None):
        """Learn the whitening transformm and cluster centroids from the example
        images.

        Parameters:

            images : iterable of 2D ndarrays
                Images to extract patches for learning from.

            n_images : int
                Number of example images provided.
            """
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
        # Optimise: multiply the whitening transform into the dictionary.
        self.cluster.centroids = _combine_proj_whiten(self.cluster.centroids,
                                                      self.whiten.whiten,
                                                      self.levels)

    def preprocess(self, image):
        """Return normalised and whitened patches extracted from the given image.
        """
        image = np.asfarray(image)
        patches = extract_patches_2d(image,
                                         patch_size=(self.pixels, self.pixels))
        patches = patches.reshape((-1, self.pixels**2))
        normalise_inplace(patches, self.whiten_reg)
        return patches

    def predict(self, images, y=None, pool=False):
        """Compute the histogram of visual word counts, over all input images.

        Parameters:
            images : iterable of 2D ndarrays
                Images to determine the histograms visual word counts from.

            pool : bool, default: False
                If True, sum the encodings for each augmented version of an 
                image, otherwise treat each augmented version as a separate 
                example.

        """
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
        """Determine the visual word encoding for every pixel in the input.

        Sum the result over all of the augmented versions of the image.

        """
        predicted = []
        i = 0
        total_words = self.n_words**self.levels
        augmented = self.augment_(image)
        encoded = [self.transform(im, reshape=True) for im in augmented]
        encoded = sum(self.augment_.inverse_transform(encoded))
        return encoded

    def transform(self, image, y=None, reshape=False):
        """Transform a single image to per pixel encoding.

        Parameters:

            image : 2D ndarray

            reshape : bool, default: False
                If True, reshape the output to match the input image size, less
                the borders.

        """
        
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

    def __call__(self, images, pool=False):
        return self.predict(images, pool=pool)


