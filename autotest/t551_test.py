"""
Test model grid class
"""
import time
import numpy as np
from flopy.grid import StructuredModelGrid
from nose.tools import timed, with_setup


class TestLargeStructuredModelGrid():

    @classmethod
    @timed(10)
    def setup_class(self):
        self.mg = StructuredModelGrid(delc=np.ones(1000), delr=np.ones(1000),
                                      xll=10, yll=10, rotation=20,
                                      epsg=26715, # coordinate reference for UTM meters, zone 15 N
                                      lenuni=1, # model grid in feet
                                      )
        print('running setup')
        assert self.mg._vertices is None
        vrts = self.mg.vertices

    @timed(1)
    def test_transform(self):
        x, y = self.mg.sr.transform(self.mg.xcenters, self.mg.ycenters)
        assert np.array_equal(x, self.mg.sr.xcenters)
        assert np.array_equal(y, self.mg.sr.ycenters)

        x, y = self.mg.sr.transform(self.mg.xedges, self.mg.yedges)
        assert np.array_equal(x, self.mg.sr.xedges)
        assert np.array_equal(y, self.mg.sr.yedges)

        x, y = self.mg.vertices1d
        assert np.array_equal(x[:5], self.mg.vertices[0, 0, :, 0])
        assert np.array_equal(y[:5], self.mg.vertices[0, 0, :, 1])
        x, y = self.mg.sr.transform(x, y)
        assert np.array_equal(x, self.mg.sr.vertices1d[:, 0])
        assert np.array_equal(y, self.mg.sr.vertices1d[:, 1])

    @timed(.1)
    def test_vertices(self):
        assert self.mg._vertices is not None
        vrts = self.mg.vertices
        assert vrts.shape == (self.mg.nrow, self.mg.ncol, 5, 2)
        assert np.array_equal(vrts[-1, 0],  np.array([[0., 1.],
                                                      [0., 0.],
                                                      [1., 0.],
                                                      [1., 1.],
                                                      [0., 1.]]))

    @timed(2)
    def test_get_vertices(self):
        for i in range(self.mg.nrow):
            for j in range(self.mg.ncol):
                self.mg.get_vertices(i, j)

    @timed(1)
    def test_get_all_vertices(self):
        assert self.mg._vertices is not None
        i, j = np.indices((self.mg.nrow, self.mg.ncol))
        self.mg.get_vertices(i, j)

    def rotate_20(self):
        print('tweaking sr')
        self.mg.sr.rotation = 45.

    def reset_rotation(self):
        print('tweaking sr')
        self.mg.sr.rotation = 20.

    @with_setup(rotate_20, reset_rotation)
    def test_rotated(self):
        print('testing transform with rotation')
        print(self.mg.sr.rotation)
        self.test_transform()


    @classmethod
    def teardown(self):
        pass