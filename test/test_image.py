import numpy as np
from BOF import BagOfFeaturesEncoder
from BOF.imageencoding import collect_normalised_patches

np.random.seed(100)

class testPatchExtractor():
    def setUp(self):
        self.images = (np.random.randint(0, high=255, size=(50,50)) for i in range(10))

    def test_collection(self):
        patches = collect_normalised_patches(self.images, 10)
        assert patches.shape == (1000, 49)
        mag = np.linalg.norm(patches, axis=1)
        assert np.logical_and(mag > 0, mag <= 1).all()

class testBOF():
    def setUp(self):
        """Generate series of random images for sampling."""
        self.images = (np.random.randint(0, high=255, size=(50,50)) for i in range(10))
        self.bof = BagOfFeaturesEncoder(n_patches=100)
        self.bof.fit(self.images, n_images=10)
        self.hier = BagOfFeaturesEncoder(n_patches=10, levels=2)
        self.hier.fit(self.images, n_images=10)
        self.test_images = (np.random.rand(50, 50) for i in range(2))

    def test_centroids(self):
        assert self.bof.cluster.centroids.shape == (10, 49)
        assert np.isfinite(self.bof.cluster.centroids).all()

    def test_transform(self):
        output = self.bof.transform(self.test_images)
        assert len(output) == 2
        assert output[0].shape == (44, 44, 10) # Patches are lost at the boundaries

    def test_transform_noreshape(self):
        output = self.bof.transform(self.test_images, reshape=False)
        assert len(output) == 2
        assert output[0].shape == (44*44, 10) # Patches are lost at the boundaries

    def test_predict(self):
        output = self.bof.predict(self.test_images)
        assert output.shape == (2, 10)
        assert np.logical_and(output[0] <= 1, output[0] >= -1).all()

    def test_hier_predict(self):
        output = self.hier.predict(self.test_images)
        assert output.shape == (2, 100)
        assert np.logical_and(output[0] <= 100, output[0] >= 0).all()

    def test_hier_transform(self):
        output = self.hier.transform(self.test_images)
        assert output[0].shape == (44*44, 2)
        assert np.logical_and(output[0] < 10, output[0] >= 0).all()

