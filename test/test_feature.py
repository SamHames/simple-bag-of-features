import numpy as np
from BOF.featurelearning import normalise_inplace, simple_cov, Whiten

np.random.seed(42)

class testNormalise():
    def setUp(self):
        self.x1 = np.arange(100).reshape((20, 5)).astype('float')
        self.x2 = self.x1.copy()

    def test_default(self):
        normalise_inplace(self.x1)
        assert np.allclose(self.x1.mean(axis=1), 0)
        assert np.allclose(self.x1.var(axis=1), 1)

    def test_regularisation(self):
        normalise_inplace(self.x1)
        normalise_inplace(self.x2, variance_reg=1)
        assert (self.x1.var(axis=1) > self.x2.var(axis=1)).all()

    def test_brightness(self):
        normalise_inplace(self.x1, brightness=False)
        assert np.allclose(self.x1.var(axis=1), 1)
        assert (self.x2.mean(axis=1) > 0).all()

    def test_avoid_copy(self):
        normalise_inplace(self.x1)
        normalise_inplace(self.x2, avoid_copy=True)
        assert (self.x1 == self.x2).all()


class test_simplecov():
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

def energy_concentration(X):
    """ Test if whitening has concentrated variance on the diagonal. """
    variances = np.abs(np.diag(X)).sum()
    upper_tri = np.abs(np.triu(X, k=1)).sum()
    return variances > upper_tri



class test_Whiten():
    def setUp(self):
        eigenvectors = np.random.rand(5,5)
        weights = np.random.rand(10,5)**2
        self.x1 = np.dot(weights, eigenvectors.T)
        self.whiten = Whiten(energy=1.0, whiten_reg=0)
        self.whiten.fit(self.x1)
        y = self.whiten.transform(self.x1)
        self.cov_full = np.cov(y.T)

    def test_eigen(self):
        assert energy_concentration(self.cov_full)
        assert np.allclose(np.diag(self.cov_full), 1)

    def test_energy(self):
        whiten_partial = Whiten(energy=0.9, whiten_reg=0)
        whiten_partial.fit(self.x1)
        cov_partial = np.cov(whiten_partial.transform(self.x1).T)
        assert energy_concentration(cov_partial)
        assert (np.diag(self.cov_full) > np.diag(cov_partial)).all()

    def test_regularisation(self):
        whiten_reg = Whiten(energy=1.0, whiten_reg=0.1)
        whiten_reg.fit(self.x1)
        cov_reg = np.cov(whiten_reg.transform(self.x1).T)
        assert energy_concentration(cov_reg)
        assert (np.diag(self.cov_full) > np.diag(cov_reg)).all()

    def test_inplace(self):
        self.whiten.transform(self.x1, inplace=True)
        cov_inplace = np.cov(self.x1.T)
        assert np.allclose(cov_inplace, self.cov_full).all()



