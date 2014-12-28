from __future__ import division, print_function
import numpy as np
from BOF.featurelearning import normalise_inplace, simple_cov, _init_random_selection
from BOF import Whiten, SphericalKMeans, HierSKMeans

np.random.seed(42)

class testNormalise():
    def setUp(self):
        self.x1 = np.arange(100).reshape((20, 5)).astype('float')
        self.x2 = self.x1.copy()

    def test_default(self):
        normalise_inplace(self.x1)
        assert np.allclose(self.x1.mean(axis=1), 0)
        assert np.allclose(np.linalg.norm(self.x1, axis=1), 1)

    def test_regularisation(self):
        normalise_inplace(self.x1)
        normalise_inplace(self.x2, norm_reg=1)
        assert (self.x1.var(axis=1) > self.x2.var(axis=1)).all()

    def test_brightness(self):
        normalise_inplace(self.x1, brightness=False)
        assert np.allclose(np.linalg.norm(self.x1, axis=1), 1)
        assert (self.x2.mean(axis=1) > 0).all()

    def test_avoid_copy(self):
        normalise_inplace(self.x1)
        normalise_inplace(self.x2, avoid_copy=True)
        assert (self.x1 == self.x2).all()


class testSimplecov():
    def setUp(self):
        self.x1 = np.arange(100).reshape((20,5)).astype('float')**2
        # Square to avoid correlated variables after normalisation
        normalise_inplace(self.x1)
        self.numpy_cov = np.cov(self.x1.T)

    def test_equivalence(self):
        cov = simple_cov(self.x1)
        assert np.allclose(cov, self.numpy_cov)

    def test_shape(self):
        cov = simple_cov(self.x1)
        assert cov.shape == (5, 5)


def energy_concentrated(X):
    """ Test if whitening has concentrated variance on the diagonal. """
    variances = np.abs(np.diag(X)).sum()
    upper_tri = np.abs(np.triu(X, k=1)).sum()
    return variances > upper_tri


class testWhiten():
    def setUp(self):
        eigenvectors = np.random.rand(5,5)
        weights = np.random.rand(10,5)**2
        self.x1 = np.dot(weights, eigenvectors.T)
        self.whiten = Whiten(energy=1.0, whiten_reg=0)
        self.whiten.fit(self.x1)
        y = self.whiten.transform(self.x1)
        self.cov_full = np.cov(y.T)

    def test_eigen(self):
        assert np.allclose(np.diag(self.cov_full), 1)

    def test_energy(self):
        whiten_partial = Whiten(energy=0.9, whiten_reg=0)
        # Normalising affects the eigenvalues
        normalise_inplace(self.x1)
        whiten_partial.fit(self.x1)
        cov_partial = np.cov(whiten_partial.transform(self.x1).T)
        assert (np.diag(self.cov_full) > np.diag(cov_partial)).all()
        assert np.isfinite(cov_partial).all()

    def test_regularisation(self):
        whiten_reg = Whiten(energy=1.0, whiten_reg=0.1)
        whiten_reg.fit(self.x1)
        cov_reg = np.cov(whiten_reg.transform(self.x1).T)
        assert (np.diag(self.cov_full) > np.diag(cov_reg)).all()

    def test_inplace(self):
        self.whiten.transform(self.x1, inplace=True)
        cov_inplace = np.cov(self.x1.T)
        assert np.allclose(cov_inplace, self.cov_full).all()

def gen_clusters(n_samples, n_dims):
    """Generate randomly perturbed cluster centers for kmeans testing."""
    centroids = np.diag(np.ones(n_dims)*100) # cardinal directions in unit hypersphere
    repeat = np.ceil(n_samples/n_dims)
    samples = np.tile(centroids, (repeat, 1))
    samples += np.random.standard_normal(samples.shape)
    return samples[:n_samples, :]

class testSphericalKMeans():
    def setUp(self):
        self.X = gen_clusters(100, 20)
        self.X1 = gen_clusters(100, 20)
        normalise_inplace(self.X)
        normalise_inplace(self.X1)
        self.skmeans = SphericalKMeans()
        self.skmeans.fit(self.X)

    def test_centroids(self):
        assert self.skmeans.centroids.shape == (10, 20)
        assert np.allclose(np.linalg.norm(self.skmeans.centroids, axis=1), 1)
        assert np.isfinite(self.skmeans.centroids).all()

    def test_transform(self):
        tform = self.skmeans.transform(self.X1)
        assert tform.shape == (100, 10)
        assert tform.max() <= 1.0 and tform.min() >= -1.0

    def test_predict(self):
        predict = self.skmeans.predict(self.X1)
        assert predict.shape == (100,)
        assert np.issubdtype(predict.dtype, np.integer)
        assert predict.max() <= 9 and predict.min() >= 0


class testHierSKMeans():
    def setUp(self):
        # Testing on small quantity of data is likely to lead to zombie clusters
        self.X = gen_clusters(100, 20)
        self.X1 = gen_clusters(100, 20)
        normalise_inplace(self.X)
        normalise_inplace(self.X1)

        self.hkmeans = HierSKMeans(n_clusters=2, levels=3)
        self.hkmeans.fit(self.X)

    def test_centroids(self):
        assert self.hkmeans.centroids[0].shape == (2, 20)
        for i in range(1, self.hkmeans.levels):
            this_layer = self.hkmeans.centroids[i]
            assert len(this_layer) == 3
            assert len(this_layer[1]) == 2

    def test_transform(self):
        tform = self.hkmeans.transform(self.X1)
        assert tform.shape == (100, self.hkmeans.levels)
        assert np.issubdtype(tform.dtype, np.integer)
        assert tform.max() <= 1 and tform.min() >= 0

    def test_predict(self):
        predict = self.hkmeans.predict(self.X1)
        assert predict.shape == (100,)
        assert np.issubdtype(predict.dtype, np.integer)
        assert predict.max() <= 7 and predict.min() >= 0




