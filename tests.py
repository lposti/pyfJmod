__author__ = 'lposti'

import unittest
from fJmodel.fJmodel import FJmodel, Potential
from numpy import array, ones
from numpy.testing import assert_almost_equal, assert_equal

# define global FJmodel
f = FJmodel("fJmodel/examples/Hernq_1.00_2.00_1.00_2.00_0.flt.out")


class MyTests(unittest.TestCase):

    def testFJ_sanity(self):

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

        trueValue = array([1., -0.5, 0.375, -0.3125])
        assert_almost_equal(f._evenlegend(0), trueValue, decimal=4)
        assert_almost_equal(f._evenlegend(1), ones(f.npoly), decimal=4)

    def testFJ_static_methods(self):

        assert_equal(f._evenlegend(0), FJmodel.even_Legendre(0, f.npoly))
        assert_equal(f._interp_pot(f.ar[0], f.phil),
                     FJmodel.interpolate_potential(f.ar[0], f.phil, f.ar, f.npoly))
        assert_equal(f._gaussLeg(0, 1), FJmodel.gauleg(0, 1, f.ngauss))

    def testPot_sanity(self):

        p = Potential(f)

        assert [x for x in p.phil[:, 0] if x <= 0]
        assert [x for x in p.ar if x > 0]
        assert [y for y in p(p.ar, p.ar[0]) if y < 0]
        assert [y for y in p(p.ar[0], p.ar) if y < 0]

    def testPot_init(self):

        p1 = Potential(fJ=f)
        p2 = Potential(phil=f.phil, Pr=f.Pr, Pr2=f.Pr2, ar=f.ar, nr=f.nr, npoly=f.npoly, ngauss=f.ngauss)

        assert_equal(p1.ar, p2.ar)
        assert_equal(p1(p1.ar, p1.ar), p2(p2.ar, p2.ar))
        assert_equal(p1.dR(p1.ar, p1.ar), p2.dR(p2.ar, p2.ar))
        assert_equal(p1.dz(p1.ar, p1.ar), p2.dz(p2.ar, p2.ar))

if __name__ == "__main__":
    unittest.main()