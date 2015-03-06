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
from numpy import fromstring, zeros, searchsorted, sqrt, asarray, ndarray


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

    def _getq(self, R, z, ql, npot = True):
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
                r = sqrt(R*R+z*z)
                c = z/r

                pol = self._evenlegend(c)
                if npot:
                    qp = self._interp(r, ql)

                    q = .5*qp[0]
                    for i in range(self.npoly):
                        f = .5*(4*i+1)
                        q += f*qp[i]*pol[i]
                else:
                    qp = self._interp_pot(r, ql)

                    q = qp[0]
                    for i in range(self.npoly):
                        q += qp[i]*pol[i]

                return q

            elif R.size > 1 and z.size == 1:
                q, r, c = zeros(R.size), zeros(R.size), zeros(R.size)
                for k in range(R.size):
                    r[k] = sqrt(R[k]*R[k]+z*z)
                    c[k] = z/r[k]

                    pol = self._evenlegend(c[k])
                    if npot:
                        qp = self._interp(r[k], ql)

                        q[k] = .5*qp[0]
                        for i in range(self.npoly):
                            f = .5*(4*i+1)
                            q[k] += f*qp[i]*pol[i]
                    else:
                        qp = self._interp_pot(r[k], ql)

                        q[k] = qp[0]
                        for i in range(self.npoly):
                            q[k] += qp[i]*pol[i]

                return q
            elif R.size == 1 and z.size > 1:
                q, r, c = zeros(z.size), zeros(z.size), zeros(z.size)
                for j in range(z.size):
                    r[j] = sqrt(R*R+z[j]*z[j])
                    c[j] = z[j]/r[j]

                    pol = self._evenlegend(c[j])
                    if npot:
                        qp = self._interp(r[j], ql)

                        q[j] = .5*qp[0]
                        for i in range(self.npoly):
                            f = .5*(4*i+1)
                            q[j] += f*qp[i]*pol[i]
                    else:
                        qp = self._interp_pot(r[j], ql)

                        q[j] = qp[0]
                        for i in range(self.npoly):
                            q[j] += qp[i]*pol[i]

                return q
            else:
                q, r, c = zeros((R.size, z.size)), zeros((R.size, z.size)), zeros((R.size, z.size))
                for k in range(R.size):
                    for j in range(z.size):
                        r[k, j] = sqrt(R[k]*R[k]+z[j]*z[j])
                        c[k, j] = z[j]/r[k, j]

                        pol = self._evenlegend(c[k, j])
                        if npot:
                            qp = self._interp(r[k, j], ql)

                            q[k, j] = .5*qp[0]
                            for i in range(self.npoly):
                                f = .5*(4*i+1)
                                q[k, j] += f*qp[i]*pol[i]
                        else:
                            qp = self._interp_pot(r[k, j], ql)

                            q[k, j] = qp[0]
                            for i in range(self.npoly):
                                q[k, j] += qp[i]*pol[i]

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
        Private method: computes the even Legendre polynomials at cos(theta)
        :param c: cos(theta), is z/sqrt(R^2+z^2) in cylindrical
        :return:  list of npoly Legendre polynomials
        """
        c2 = c*c
        pol = zeros(self.npoly, dtype=float)

        pol[0] = 1
        if self.npoly < 2:
            return

        pol[1] = 1.5*c2-.5
        for np in range(2, self.npoly):
            l = 2*(np-1)
            l2 = 2*l
            pol[np] = -pol[np-2]*l*(l-1)/float((l2+1)*(l2-1)) + \
                pol[np-1]*(c2-(l2*l+l2-1)/float((l2-1)*(l2+3)))
            pol[np] *= (l2+1)*(l2+3)/float((l+1)*(l+2))

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
            bot = searchsorted(self.ar, r, side='left')-1
            top = bot+1

            f = (r-self.ar[bot])/(self.ar[top]-self.ar[bot])
            for i in range(self.npoly):
                intp[i] = f*ql[top][i]+(1-f)*ql[bot][i]

        return intp

    def _interp_pot(self, r, phil):

        phip = zeros(self.npoly, dtype=float)

        if r > self.ar[-1]:
            for k in range(self.npoly):
                phip[k] = phil[-1][k]*pow(self.ar[-1]/r, 2*k+1)

        else:
            bot = searchsorted(self.ar, r, side='left')-1
            top = bot+1

            db = r-self.ar[bot]
            f1 = db/(self.ar[top]-self.ar[bot])

            for k in range(self.npoly):
                phip[k] = f1*phil[top][k]+(1-f1)*phil[bot][k]

            if top < 10:
                if f1 < 0.5 and bot > 0:
                    thr = bot-1
                else:
                    thr = top+1

                dt = r-self.ar[top]
                f2 = dt*db/((self.ar[thr]-self.ar[top])*(self.ar[thr]-self.ar[bot]))
                f3 = (self.ar[thr]-self.ar[bot])/(self.ar[top]-self.ar[bot])

                for k in range(self.npoly):
                    phip[k] += f2*(phil[thr][k]-phil[bot][k]-f3*(phil[top][k]-phil[bot][k]))

        return phip