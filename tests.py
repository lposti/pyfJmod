__author__ = 'lposti'

import unittest
from fJmodel.fJmodel import FJmodel, Potential
from numpy import array, ones, zeros
from numpy.testing import assert_almost_equal, assert_equal

# define global FJmodel
f = FJmodel("fJmodel/examples/Hernq_0.55_0.55_1.00_1.00_4.out")


class MyTests(unittest.TestCase):

    def testFJ_sanity(self):

        # Assertion list
        # for some I use list comprehension to compact the code
        assert f.nr > 0 and f.npoly > 0 and f.ngauss > 0
        assert [x for x in f.ar if x > 0]
        assert [x for x in f.rhl[:, 0] if x > 0]
        assert [x for x in f.vrotl[:, 0] if x > 0]
        assert [x for x in f.sigRl[:, 0] if x > 0]
        assert [x for x in f.sigpl[:, 0] if x > 0]
        assert [x for x in f.sigzl[:, 0] if x > 0]
        assert [x for x in f.sigRzl[:, 0] if x > 0]
        for x in f.ar:
            assert f.rho(x, 0) > 0
        assert [y for y in f.rho(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.sigR(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.sigp(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.sigz(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.sigRz(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.vrot(f.ar, f.ar[0]) if y > 0]
        assert [y for y in f.rho(f.ar[0], f.ar) if y > 0]
        assert [y for y in f.sigR(f.ar[0], f.ar) if y > 0]
        assert [y for y in f.sigp(f.ar[0], f.ar) if y > 0]
        assert [y for y in f.sigz(f.ar[0], f.ar) if y > 0]
        assert [y for y in f.sigRz(f.ar[0], f.ar) if y > 0]
        assert [y for y in f.vrot(f.ar[0], f.ar) if y > 0]

    def testFJ_Legendre(self):

        trueValue = array([1., -0.5, 0.375, -0.3125])
        assert_almost_equal(f._evenlegend(0), trueValue, decimal=4)
        assert_almost_equal(f._evenlegend(1), ones(f.npoly), decimal=4)

    def testFJ_static_methods(self):

        assert_equal(f._evenlegend(0), FJmodel.even_Legendre(0, f.npoly))
        assert_equal(f._interp_pot(f.ar[0], f.phil),
                     FJmodel.interpolate_potential(f.ar[0], f.phil, f.ar, f.npoly))
        assert_equal(f._gaussLeg(0, 1), FJmodel.gauleg(0, 1, f.ngauss))

    def test_virialized_output(self):

        assert_almost_equal(f.virial(verbose=False, ret=True), (-2., -2., -2.), decimal=0)

    def test_final_mass(self):

        assert_almost_equal(f.compare_mass(verbose=False), 1., decimal=1)

    def test_ellipticity_spherical_model(self):

        assert_almost_equal(f.eps, zeros(len(f.eps), dtype=float), decimal=1)

    def test_project_spherical_model(self):

        f.project(inclination=90, nx=31, npsi=31, verbose=False)
        half = f.nr / 2
        delta = f.dlos[half, half:] - f.dlos[half:, half]
        assert_almost_equal(delta, zeros(len(delta), dtype=float), decimal=1)

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