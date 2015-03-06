__author__ = 'lposti'

import unittest
from fJmodel import FJmodel
from numpy import array, ones
from numpy.testing import assert_almost_equal


class MyTests(unittest.TestCase):

    def testFJ_sanity(self):
        f = FJmodel("/Users/lp1osti/git_fJmodels/models/Hernq_0.55_0.55_1.00_1.00_0.out")

        # Assertion list
        # for some I use list comprehension to compact the code
        assert f.nr > 0 and f.npoly > 0 and f.ngauss > 0
        assert [x for x in f.ar if x > 0]
        assert [x for x in f.rhl[:, 0] if x > 0]
        assert [x for x in f.sigRl[:, 0] if x > 0]
        assert [x for x in f.sigpl[:, 0] if x > 0]
        assert [x for x in f.sigzl[:, 0] if x > 0]
        assert [y for y in f.rho(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.sigR(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.sigp(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.sigz(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.rho(f.ar[0], f.ar) if y > 0]
        assert [y for y in f.sigR(f.ar[0], f.ar) if y > 0]
        assert [y for y in f.sigp(f.ar[0], f.ar) if y > 0]
        assert [y for y in f.sigz(f.ar[0], f.ar) if y > 0]

    def testFJ_Legendre(self):
        f = FJmodel("/Users/lp1osti/git_fJmodels/models/Hernq_0.55_0.55_1.00_1.00_0.out")

        trueValue = array([1., -0.5,  0.375, -0.3125])
        assert_almost_equal(f._evenlegend(0), trueValue, decimal=4)
        assert_almost_equal(f._evenlegend(1), ones(f.npoly), decimal=4)

if __name__ == "__main__":
    unittest.main()