import numpy as np
from BOF import BagOfFeaturesEncoder
from BOF.imageencoding import collect_normalised_patches, Augment

np.random.seed(10)


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
        self.test_images = [np.random.rand(50, 50) for i in range(2)]

    def test_centroids(self):
        assert self.bof.cluster.centroids.shape == (10, 49)
        assert np.isfinite(self.bof.cluster.centroids).all()

    def test_transform_reshape(self):
        output = self.bof.transform(self.test_images[0], reshape=True)
        assert output.shape == (44, 44, 10) # Patches are lost at the boundaries
        assert output.dtype =='bool'

    def test_transform_noreshape(self):
        output = self.bof.transform(self.test_images[0], reshape=False)
        assert output.shape == (44*44, ) # Patches are lost at the boundaries

    def test_predict(self):
        output = self.bof.predict(self.test_images)
        assert output.shape == (2, 10)
        assert (output.sum(axis=1) == 44*44).all()

    def test_hier_predict(self):
        output = self.hier.predict(self.test_images)
        assert output.shape == (2, 100)
        assert (output.sum(axis=1) == 44*44).all()

    def test_hier_transform(self):
        output = self.hier.transform(self.test_images[0])
        assert output.shape == (44*44,)
        assert np.logical_and(output < 100, output >= 0).all()

    def test_predict_pixels(self):
        prediction = self.bof.predict_pixels(self.test_images[0])
        assert prediction.shape == (44, 44, 10)
    # Still need to test the augmented representation and pooling outputs.


class testAugment():
    def setUp(self):
        self.image = np.random.rand(40,30)
    
    def test_rotate(self):
        augment = Augment('rotate')
        augmented = augment(self.image)
        assert len(augmented) == 4
        assert augmented[0].shape == augmented[2].shape == (40, 30)
        assert augmented[1].shape == augmented[3].shape == (30, 40)
        assert (np.rot90(augmented[-1]) == self.image).all

    def test_reflect(self):
        augment = Augment('reflect')
        augmented = augment(self.image)
        assert len(augmented) == 3
        assert augmented[0].shape == augmented[1].shape == augmented[2].shape == (40, 30)
        assert (np.flipud(augmented[2]) == self.image).all()
        assert (np.fliplr(augmented[1]) == self.image).all()

    def test_both(self):
        augment = Augment('both')
        augmented = augment(self.image)
        assert len(augmented) == 12
