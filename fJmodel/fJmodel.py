#################################################
#
#  fJmodel class
#
#  Author:  L. Posti (lorenzo.posti@gmail.com)
#  Commits: 02/03/15 Class creation
#
##################################################
__author__ = 'lposti'


from os.path import isfile
from linecache import getline
from numpy import fromstring, zeros, searchsorted, sqrt, asarray, ndarray, cos, pi, arccos, inf, trapz
from scipy.integrate import tplquad


class FJmodel(object):
    """
    Class for handling f(J) model data
    """
    def __init__(self, filename):
        """
        Constructor of the class
        :param filename: full path of the datafile
        :return: Initializes the class with the arrays; ar, rhl, vrotl, sigRl, sigpl, sigzl, sigRzl
        """
        try:
            self.fname = filename
            assert isfile(self.fname) is True

            self._line = 1
            while getline(self.fname, self._line)[0] == '#':
                self._line += 1

            self.nr, self.npoly, self.ngauss = fromstring(getline(self.fname, self._line), dtype=int, sep=' ')
            self._line += 1

            self.ar = self._getr()
            self.rhl = self._getLeg()
            self.vrotl = self._getLeg()
            self.sigRl = self._getLeg()
            self.sigpl = self._getLeg()
            self.sigzl = self._getLeg()
            self.sigRzl = self._getLeg()
            self.phil = self._getLeg()
            self.Pr = self._getLeg()
            self.Pr2 = self._getLeg()

            # compute spherically averaged Mass
            self.mass = 4. * pi * trapz(self.ar * self.ar * self.rho(self.ar, 0), self.ar)

        except AssertionError:
            print "Assert Error: file does not exist!!"

    def phi(self, R, z):
        """
        API to get the potential of the given f(J) model.
        :param R: Cylindrical radius (array type)
        :param z: Cylindrical height on equatorial plane (array type)
        :return:  Potential of the model at location in meridional plane (array type)
        """
        R, z = asarray(R), asarray(z)
        return self._getq(R, z, self.phil, npot=False)

    def rho(self, R, z):
        """
        API to get density of the given f(J) model.
        :param R: Cylindrical radius (array type)
        :param z: Cylindrical height on equatorial plane (array type)
        :return:  Density of the model at location in meridional plane (array type)
        """
        R, z = asarray(R), asarray(z)
        return self._getq(R, z, self.rhl)

    def vrot(self, R, z):
        """
        API to get rotational velocity of the given f(J) model.
        :param R: Cylindrical radius (array type)
        :param z: Cylindrical height on equatorial plane (array type)
        :return:  Rotation velocity of the model at location in meridional plane (array type)
        """
        R, z = asarray(R), asarray(z)
        return self._getq(R, z, self.vrotl)

    def sigR(self, R, z):
        """
        API to get radial velocity dispersion of the given f(J) model.
        :param R: Cylindrical radius (array type)
        :param z: Cylindrical height on equatorial plane (array type)
        :return:  Radial velocity dispersion of the model at location in meridional plane (array type)
        """
        R, z = asarray(R), asarray(z)
        return self._getq(R, z, self.sigRl)

    def sigp(self, R, z):
        """
        API to get azimuthal velocity dispersion of the given f(J) model.
        :param R: Cylindrical radius (array type)
        :param z: Cylindrical height on equatorial plane (array type)
        :return:  Azimuthal velocity dispersion of the model at location in meridional plane (array type)
        """
        R, z = asarray(R), asarray(z)
        return self._getq(R, z, self.sigpl)

    def sigz(self, R, z):
        """
        API to get vertical velocity dispersion of the given f(J) model.
        :param R: Cylindrical radius (array type)
        :param z: Cylindrical height on equatorial plane (array type)
        :return:  Vertical velocity dispersion of the model at location in meridional plane (array type)
        """
        R, z = asarray(R), asarray(z)
        return self._getq(R, z, self.sigzl)

    def sigRz(self, R, z):
        """
        API to get off-diagonal term in the velocity dispersion tensor of the f(J) model.
        :param R: Cylindrical radius (array type)
        :param z: Cylindrical height on equatorial plane (array type)
        :return:  Off-diagonal velocity dispersion term of the model at location in meridional plane (array type)
        """
        R, z = asarray(R), asarray(z)
        return self._getq(R, z, self.sigRzl)

    def virial(self):
        """
        Computes and prints the quantity involved in the tensor virial theorem.
        We judge a system to be relaxed to self-consistency if the ratios of the potential over
        kinetic energies (diagonal and off diagonal tensor terms) are reasonably close to 2
        :return: Prints the statistic for the tensor virial theorem
        """
        p = Potential(fJ=self)
        ci, wi = self._gaussLeg(0, 1)

        pot, KRR, Kzz, WRR, Wzz = 0, 0, 0, 0, 0
        for i in range(1, self.nr):
            r, dr = .5 * (self.ar[i - 1] + self.ar[i]), self.ar[i] - self.ar[i - 1]
            for j in range(self.ngauss):
                sij = sqrt(1. - ci[j] * ci[j])
                R, z = r * sij, r * ci[j]

                dens = self.rho(R, z)
                pot += .5 * wi[j] * dens * self.phi(R, z) * r * r * dr
                KRR += .5 * wi[j] * dens * (pow(r * self.sigR(R, z), 2) + pow(r * self.sigp(R, z), 2)) * dr
                Kzz += .5 * wi[j] * dens * pow(r * self.sigz(R, z), 2) * dr
                WRR -= wi[j] * dens * R * p.dR(R, z) * r * r * dr
                Wzz -= wi[j] * dens * z * p.dz(R, z) * r * r * dr

        KRR *= 4 * pi
        Kzz *= 4 * pi
        WRR *= 4 * pi
        Wzz *= 4 * pi
        pot *= 4 * pi

        print "  Virial statistic: "
        print "  Mass(%3.1f): %f %f" % (self.ar[-1], pow(self.ar[-1], 2) * self.Pr[-1, 0], self.mass)
        print "  KE, PE, W/K = %f %f %f " % (KRR + Kzz, pot, pot / (KRR + Kzz))
        print "  Kxx, Wxx, Wxx/Kxx = %f %f %f" % (KRR, WRR, WRR / KRR)
        print "  Kzz, Wzz, Wzz/Kzz = %f %f %f" % (Kzz, Wzz, Wzz / Kzz)

    def compareMass(self, alpha=1.667, beta=5., M0=1., r0=1., dphih=.55, dzh=.55, dphig=.55, dzg=.55):

        J0 = sqrt(M0 * r0)
        h = lambda Jr, Jphi, Jz: Jr + dphih * Jphi + dzh * Jz
        g = lambda Jr, Jphi, Jz: Jr + dphig * Jphi + dzg * Jz
        DF = lambda Jr, Jphi, Jz: M0 / pow(J0, 3) * pow(1. + J0 / h(Jr, Jphi, Jz), alpha) /\
            (pow(1 + g(Jr, Jphi, Jz) / J0, beta))
        massfJ = tplquad(DF, 0, inf, lambda x: 0, lambda x: inf,
                         lambda x, y: 0, lambda x, y: inf)[0]

        print "Mass: f(J) model = %f, eq. (37)=%f" % (self.mass, massfJ)

    def _getq(self, R, z, ql, npot=True):
        """
        Private method: gets the intended quantity interpolating the Legendre coefficients
        ql in the location (R, z) in the meridional plane
        :param R:  Cylindrical radius (array type)
        :param z:  Cylindrical height on equatorial plane (array type)
        :param ql: Legendre coefficients for the intended quantity (double array type with shape nr, npoly)
        :return:   The desired quantity interpolated at (R, z). Is a scalar, array or double array dep. on R, z
        """
        try:
            assert type(R) is ndarray
            assert type(z) is ndarray
            if R.size == 1 and z.size == 1:
                r = sqrt(R * R + z * z)
                c = z / r

                pol = self._evenlegend(c)
                if npot:
                    qp = self._interp(r, ql)

                    q = .5 * qp[0]
                    for i in range(1, self.npoly):
                        f = .5 * (4 * i + 1)
                        q += f * qp[i] * pol[i]
                else:
                    qp = self._interp_pot(r, ql)

                    q = qp[0]
                    for i in range(1, self.npoly):
                        q += qp[i] * pol[i]

                return q

            elif R.size > 1 and z.size == 1:
                q, r, c = zeros(R.size), zeros(R.size), zeros(R.size)
                for k in range(R.size):
                    r[k] = sqrt(R[k] * R[k] + z * z)
                    c[k] = z / r[k]

                    pol = self._evenlegend(c[k])
                    if npot:
                        qp = self._interp(r[k], ql)

                        q[k] = .5 * qp[0]
                        for i in range(1, self.npoly):
                            f = .5 * (4 * i + 1)
                            q[k] += f * qp[i] * pol[i]
                    else:
                        qp = self._interp_pot(r[k], ql)

                        q[k] = qp[0]
                        for i in range(1, self.npoly):
                            q[k] += qp[i] * pol[i]

                return q
            elif R.size == 1 and z.size > 1:
                q, r, c = zeros(z.size), zeros(z.size), zeros(z.size)
                for j in range(z.size):
                    r[j] = sqrt(R * R + z[j] * z[j])
                    c[j] = z[j] / r[j]

                    pol = self._evenlegend(c[j])
                    if npot:
                        qp = self._interp(r[j], ql)

                        q[j] = .5 * qp[0]
                        for i in range(1, self.npoly):
                            f = .5 * (4 * i + 1)
                            q[j] += f * qp[i] * pol[i]
                    else:
                        qp = self._interp_pot(r[j], ql)

                        q[j] = qp[0]
                        for i in range(1, self.npoly):
                            q[j] += qp[i] * pol[i]

                return q
            else:
                q, r, c = zeros((R.size, z.size)), zeros((R.size, z.size)), zeros((R.size, z.size))
                for k in range(R.size):
                    for j in range(z.size):
                        r[k, j] = sqrt(R[k] * R[k] + z[j] * z[j])
                        c[k, j] = z[j] / r[k, j]

                        pol = self._evenlegend(c[k, j])
                        if npot:
                            qp = self._interp(r[k, j], ql)

                            q[k, j] = .5 * qp[0]
                            for i in range(1, self.npoly):
                                f = .5 * (4 * i + 1)
                                q[k, j] += f * qp[i] * pol[i]
                        else:
                            qp = self._interp_pot(r[k, j], ql)

                            q[k, j] = qp[0]
                            for i in range(1, self.npoly):
                                q[k, j] += qp[i] * pol[i]

                return q
        except AssertionError:
            print "ERROR assertion of ndarray"

    def _getr(self):
        """
        Private method: read the ar array from datafile
        :return: array ar
        """
        ar = zeros(self.nr)
        for i in range(self.nr):
            ar[i] = fromstring(getline(self.fname, self._line), dtype=float, sep=' ')[0]
            self._line += 1

        return ar

    def _getLeg(self):
        """
        Private method: read a double array from datafile
        :return: double array sig
        """
        sig = zeros((self.nr, self.npoly))
        for i in range(self.nr):
            line = fromstring(getline(self.fname, self._line), dtype=float, sep=' ')
            self._line += 1

            for j in range(len(line)):
                sig[i, j] = line[j]

        return sig

    def _evenlegend(self, c):
        """
        Calls static method to compute the even Legendre polynomials at cos(theta)
        :param c: cos(theta), is z/sqrt(R^2+z^2) in cylindrical
        :return:  list of npoly Legendre polynomials
        """
        return self.even_Legendre(c, self.npoly)

    @staticmethod
    def even_Legendre(c, npoly):
        """
        Static method: gets the even Legendre polynomials at cos(theta)
        :param c: cos(theta)
        :param npoly: number of polynomials (=2*l)
        :return: list of npoly Legendre polynomials
        """
        c2 = c * c
        pol = zeros(npoly, dtype=float)

        pol[0] = 1
        if npoly < 2:
            return

        pol[1] = 1.5 * c2 - .5
        for np in range(2, npoly):
            l = 2 * (np - 1)
            l2 = 2 * l
            pol[np] = -pol[np - 2] * l * (l - 1) / float((l2 + 1) * (l2 - 1)) + \
                pol[np - 1] * (c2 - (l2 * l + l2 - 1) / float((l2 - 1) * (l2 + 3)))
            pol[np] *= (l2 + 1) * (l2 + 3) / float((l + 1) * (l + 2))

        return pol

    def _interp(self, r, ql):
        """
        Private method: interpolates the Legendre coefficients at location r
        :param r:  radius to which interpolate
        :param ql: Legendre coefficients to interpolate
        :return:   list of npoly Legendre coefficients
        """
        intp = zeros(self.npoly, dtype=float)

        if r > self.ar[-1]:
            pass
        else:
            bot = searchsorted(self.ar, r, side='left') - 1
            top = bot + 1

            f = (r - self.ar[bot]) / (self.ar[top] - self.ar[bot])
            for i in range(self.npoly):
                intp[i] = f * ql[top][i] + (1 - f) * ql[bot][i]

        return intp

    def _interp_pot(self, r, phil):
        """
        Calls static method _interpolate_potential with correct parameters
        :param r: radius to which interpolate
        :param phil: Legendre coefficients to interpolate
        :return: list of npoly Legendre coefficients
        """
        return self.interpolate_potential(r, phil, self.ar, self.npoly)

    @staticmethod
    def interpolate_potential(r, phil, ar, npoly, Pr=None, Pr2=None):
        """
        Static method: interpolates the Legendre coefficients of the potential at spherical r
        :param r: radius
        :param phil: list of Legendre coefficients
        :param ar: radial grid where the coefficients are given
        :param npoly: number of Legendre coefficients
        :param Pr: list of Legendre coefficients for phi' (Optional, if given returns also interpolated phi' coeffs.)
        :param Pr2: list of Legendre coefficients for phi'' (Optional, if given returns also interpolated phi'' coeffs.)
        :return: list of interpolated phi coeffs. (Optional: list of interpolated phi', phi'' coeffs.)
        """

        assert type(ar) is ndarray
        nr = len(ar)
        assert npoly > 0
        phip, dphip, d2phip = zeros(npoly, dtype=float), zeros(npoly, dtype=float), zeros(npoly, dtype=float)

        if r > ar[-1]:
            for k in range(npoly):
                phip[k] = phil[-1, k] * pow(ar[-1] / r, 2 * k + 1)
                if Pr is not None and nr is not None:
                    dphip[k] = - (2 * k + 1) * phil[nr - 1, k] * pow(ar[nr - 1] / r, 2 * k + 1) / r
                if Pr2 is not None and nr is not None:
                    d2phip[k] = (2 * k + 2) * (2 * k + 1) * phil[nr - 1, k] * pow(ar[nr - 1] / r, 2 * k + 1) / r / r

        else:
            bot = searchsorted(ar, r, side='left') - 1
            top = bot + 1

            db = r - ar[bot]
            f1 = db / (ar[top] - ar[bot])

            for k in range(npoly):
                phip[k] = f1 * phil[top, k] + (1 - f1) * phil[bot, k]
                if Pr is not None and nr is not None:
                    dphip[k] = f1 * Pr[top, k] + (1 - f1) * Pr[bot, k]
                if Pr2 is not None and nr is not None:
                    d2phip[k] = f1 * Pr2[top, k] + (1 - f1) * Pr2[bot, k]

            if top < 10:
                if f1 < 0.5 and bot > 0:
                    thr = bot - 1
                else:
                    thr = top + 1

                dt = r - ar[top]
                f2 = dt * db / ((ar[thr] - ar[top]) * (ar[thr] - ar[bot]))
                f3 = (ar[thr] - ar[bot]) / (ar[top] - ar[bot])

                for k in range(npoly):
                    phip[k] += f2 * (phil[thr, k] - phil[bot, k] - f3 * (phil[top, k] - phil[bot, k]))
                    if Pr is not None and nr is not None:
                        dphip[k] += f2 * (Pr[thr, k] - Pr[bot, k] - f3 * (Pr[top, k] - Pr[bot, k]))
                    if Pr2 is not None and nr is not None:
                        d2phip[k] += f2 * (Pr2[thr, k] - Pr2[bot, k] - f3 * (Pr2[top, k] - Pr2[bot, k]))

        if Pr is not None:
            if Pr2 is not None:
                return phip, dphip, d2phip
            else:
                return phip, dphip
        else:
            return phip

    def _gaussLeg(self, x1, x2):
        """
        Calls static method gauleg
        :param x1: begin
        :param x2: end
        :return: Gauss-Legendre coordinates and weights
        """
        return self.gauleg(x1, x2, self.ngauss)

    @staticmethod
    def gauleg(x1, x2, ngauss):
        """
        Static method: computes the coordinates (zeros of Leg. polynomials) and weights for Gauss-Legendre integration
        :param x1: x begin
        :param x2: x end
        :param ngauss: number of points
        :return: numpy.arrays of coordinates and weights
        """
        assert ngauss > 0
        x, w = zeros(ngauss), zeros(ngauss)
        m = (ngauss + 1) / 2
        xm, xl = 0.5 * (x2 + x1), 0.5 * (x2 - x1)
        for i in range(m):
            z, z1 = cos(pi * (i + 0.75) / (ngauss + 0.5)), 0

            while abs(z - z1) > 1e-10:
                p1, p2 = 1, 0
                for j in range(ngauss):
                    p3 = p2
                    p2 = p1
                    p1 = ((2 * (j + 1) - 1) * z * p2 - j * p3) / float(j + 1)

                pp = ngauss * (z * p1 - p2) / (z * z - 1.0)
                z1 = z
                z = z1 - p1 / pp

            x[i] = xm - xl * z
            x[ngauss - i - 1] = xm + xl * z
            w[i] = 2 * xl / ((1 - z * z) * pp * pp)
            w[ngauss - i - 1] = w[i]

        return x, w


class Potential(object):
    """
    Class for initializing a potential object to get phi, phi', phi'' at arbitrary locations
    """

    def __init__(self, fJ=None, phil=None, Pr=None, Pr2=None, ar=None, nr=None, npoly=None, ngauss=None):
        """
        Constructor: either init with a FJmodel object or by manually giving phil, ar, nr, npoly, ngauss
        :param fJ: FJmodel object
        :param phil: Leg. coeffs. for phi (Optional)
        :param Pr: Leg. coeffs. for phi' (Optional)
        :param Pr2: Leg. coeffs. for phi'' (Optional)
        :param ar: radial grid (Optional)
        :param nr: number of grid points
        :param npoly: number of Leg. coeffs.
        :param ngauss: number of Gauss comp.
        :return: initializes Potential object
        """
        if fJ is not None:
            assert type(fJ) is FJmodel
            self.fJ = fJ
            self.phil, self.Pr, self.Pr2 = fJ.phil, fJ.Pr, fJ.Pr2
            self.ar = fJ.ar
            self.nr, self.npoly, self.ngauss = fJ.nr, fJ.npoly, fJ.ngauss

        elif phil is not None and \
            ar is not None and \
            nr is not None and \
            npoly is not None and \
            ngauss is not None:

                self.phil, self.Pr, self.Pr2 = phil, Pr, Pr2
                self.ar = ar
                self.nr, self.npoly, self.ngauss = nr, npoly, ngauss

        else:
            raise ValueError("Call constructor either with FJmodel object or with phil, ar, nr, npoly, ngauss!")

    def __call__(self, R, z):
        """
        Call method overriding: gives the potential at a specified location
        :param R: cylindrical radius (array type)
        :param z: cylindrical height (array type)
        :return: Potential (array type)
        """
        R, z = asarray(R), asarray(z)

        # scalar case
        if R.size == 1 and z.size == 1:
            c, r = z / sqrt(R * R + z * z), sqrt(R * R + z * z)
            pol = FJmodel.even_Legendre(c, self.npoly)
            phip = FJmodel.interpolate_potential(r, self.phil, self.ar, self.npoly)

            phi = phip[0]
            for i in range(1, self.npoly):
                phi += phip[i] * pol[i]

        # R array case
        elif R.size > 1 and z.size == 1:
            phi = zeros(R.size)
            for k in range(R.size):
                c, r = z / sqrt(R[k] * R[k] + z * z), sqrt(R[k] * R[k] + z * z)
                pol = FJmodel.even_Legendre(c, self.npoly)
                phip = FJmodel.interpolate_potential(r, self.phil, self.ar, self.npoly)

                phi[k] = phip[0]
                for i in range(1, self.npoly):
                    phi[k] += phip[i] * pol[i]

        # z array case
        elif R.size == 1 and z.size > 1:
            phi = zeros(z.size)
            for j in range(z.size):
                c, r = z[j] / sqrt(R * R + z[j] * z[j]), sqrt(R * R + z[j] * z[j])
                pol = FJmodel.even_Legendre(c, self.npoly)
                phip = FJmodel.interpolate_potential(r, self.phil, self.ar, self.npoly)

                phi[j] = phip[0]
                for i in range(1, self.npoly):
                    phi[j] += phip[i] * pol[i]

        # [R, z] array case
        elif R.size > 1 and z.size > 1:
            phi = zeros((R.size, z.size))
            for k in range(R.size):
                for j in range(z.size):
                    c, r = z[j] / sqrt(R[k] * R[k] + z[j] * z[j]), sqrt(R[k] * R[k] + z[j] * z[j])
                    pol = FJmodel.even_Legendre(c, self.npoly)
                    phip = FJmodel.interpolate_potential(r, self.phil, self.ar, self.npoly)

                    phi[k, j] = phip[0]
                    for i in range(1, self.npoly):
                        phi[k, j] += phip[i] * pol[i]
        else:
            raise IndexError("Cannot determine size of R and/or z in Phi")

        return phi

    def dR(self, R, z):
        """
        Compute the radial derivative dPhi/dR of Phi in cylindrical coordinates
        :param R: cylindrical radius (array type)
        :param z: cylindrical height (array type)
        :return: dPhi/dR (array type)
        """
        R, z = asarray(R), asarray(z)

        # scalar case
        if R.size == 1 and z.size == 1:
            r = sqrt(z * z + R * R)
            if r < self.ar[0]:
                r = self.ar[0]

            s, c = R / r, z / r
            if c == 1:
                c -= 1e-8

            theta = arccos(c)
            dphidr = self._dtheta(r, c) * (c / r) + self._dr(r, theta) * s

        # R array case
        elif R.size > 1 and z.size == 1:
            dphidr = zeros(R.size)
            for k in range(R.size):
                r = sqrt(z * z + R[k] * R[k])
                if r < self.ar[0]:
                    r = self.ar[0]

                s, c = R[k] / r, z / r
                if c == 1:
                    c -= 1e-8

                theta = arccos(c)
                dphidr[k] = self._dtheta(r, c) * (c / r) + self._dr(r, theta) * s

        # z array case
        elif R.size == 1 and z.size > 1:
            dphidr = zeros(z.size)
            for j in range(z.size):
                r = sqrt(z[j] * z[j] + R * R)
                if r < self.ar[0]:
                    r = self.ar[0]

                s, c = R / r, z[j] / r
                if c == 1:
                    c -= 1e-8

                theta = arccos(c)
                dphidr[j] = self._dtheta(r, c) * (c / r) + self._dr(r, theta) * s

        # [R, z] array case
        elif R.size > 1 and z.size > 1:
            dphidr = zeros((R.size, z.size))
            for k in range(R.size):
                for j in range(z.size):
                    r = sqrt(z[j] * z[j] + R[k] * R[k])
                    if r < self.ar[0]:
                        r = self.ar[0]

                    s, c = R[k] / r, z[j] / r
                    if c == 1:
                        c -= 1e-8

                    theta = arccos(c)
                    dphidr[k, j] = self._dtheta(r, c) * (c / r) + self._dr(r, theta) * s
        else:
            raise IndexError("Cannot determine size of R and/or z in dPhi/dR")

        return dphidr

    def dz(self, R, z):
        """
        Compute the radial derivative dPhi/dR of Phi in cylindrical coordinates
        :param R: cylindrical radius (array type)
        :param z: cylindrical height (array type)
        :return: dPhi/dR (array type)
        """
        R, z = asarray(R), asarray(z)

        # scalar case
        if R.size == 1 and z.size == 1:
            r = sqrt(z * z + R * R)
            if r < self.ar[0]:
                r = self.ar[0]

            s, c = R / r, z / r
            if c == 1:
                c -= 1e-8

            theta = arccos(c)
            dphidz = self._dtheta(r, c) * (-s / r) + self._dr(r, theta) * c

        # R array case
        elif R.size > 1 and z.size == 1:
            dphidz = zeros(R.size)
            for k in range(R.size):
                r = sqrt(z * z + R[k] * R[k])
                if r < self.ar[0]:
                    r = self.ar[0]

                s, c = R[k] / r, z / r
                if c == 1:
                    c -= 1e-8

                theta = arccos(c)
                dphidz[k] = self._dtheta(r, c) * (-s / r) + self._dr(r, theta) * c

        # z array case
        elif R.size == 1 and z.size > 1:
            dphidz = zeros(z.size)
            for j in range(z.size):
                r = sqrt(z[j] * z[j] + R * R)
                if r < self.ar[0]:
                    r = self.ar[0]

                s, c = R / r, z[j] / r
                if c == 1:
                    c -= 1e-8

                theta = arccos(c)
                dphidz[j] = self._dtheta(r, c) * (-s / r) + self._dr(r, theta) * c

        # [R, z] array case
        elif R.size > 1 and z.size > 1:
            dphidz = zeros((R.size, z.size))
            for k in range(R.size):
                for j in range(z.size):
                    r = sqrt(z[j] * z[j] + R[k] * R[k])
                    if r < self.ar[0]:
                        r = self.ar[0]

                    s, c = R[k] / r, z[j] / r
                    if c == 1:
                        c -= 1e-8

                    theta = arccos(c)
                    dphidz[k, j] = self._dtheta(r, c) * (-s / r) + self._dr(r, theta) * c
        else:
            raise IndexError("Cannot determine size of R and/or z in dPhi/dz")

        return dphidz

    def _dr(self, r, theta):
        """
        Private method: compute derivative w.r.t. spherical r, i.e., dPhi/dr
        :param r: spherical radius (scalar)
        :param theta: angle (scalar)
        :return: dPhi/dr (scalar)
        """
        pol = FJmodel.even_Legendre(cos(theta), self.npoly)
        phip, dphip = FJmodel.interpolate_potential(r, self.phil, self.ar, self.npoly,
                                                    Pr=self.Pr)

        dr = 0.
        for i in range(self.npoly):
            dr += dphip[i] * pol[i]

        return dr

    def _dtheta(self, r, c):
        """
        Private method: compute derivative w.r.t. spherical theta, i.e., dPhi/dtheta
        :param r: spherical radius (scalar)
        :param c: cos(theta) (scalar)
        :return: dPhi/dtheta (scalar)
        """
        dtheta, dpol = 0, zeros(self.npoly)
        phip = FJmodel.interpolate_potential(r, self.phil, self.ar, self.npoly)

        for i in range(self.npoly):
            dpol[i] = self.dlegend(c, 2 * i)

        for i in range(self.npoly):
            dtheta += phip[i] * dpol[i]

        dtheta *= -sqrt(1 - c * c)
        return dtheta

    @staticmethod
    def legend(allpol, c, npoly):

        npoly += 1
        allpol[0] = 1
        if npoly < 2:
            raise ValueError("Found npoly <2!")

        allpol[1] = c
        for i in range(2, npoly):
            allpol[i] = ((2 * i - 1) * c * allpol[i - 1] - (i - 1) * allpol[i - 2]) / float(i)

    @staticmethod
    def dlegend(c, n):

        if n == 0:
            return 0
        allpol = zeros(n + 1)
        Potential.legend(allpol, c, n)
        return (n * allpol[n - 1] - n * c * allpol[n]) / (1 - c * c)
