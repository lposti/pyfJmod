# cython: profile=True
# cython: cdivision=True
# cython: nonecheck=False
__author__ = 'lposti'


from numpy import pi, zeros, logspace, concatenate, seterr, searchsorted
from numpy cimport ndarray, double_t
from progressbar import ProgressBar, widgets
import cython
from cython.parallel import prange


# C definitions
cdef extern from "math.h":
    double log10 ( double x )
    double fabs ( double x )
    double cos ( double x )
    double sin ( double x )
    double sqrt ( double x )
    double cosh ( double x )
    double sinh ( double x )
    double atan2 ( double x, double y )

cpdef projection(double incl, double b, double Rmax, int nx, int npsi, scale='linear',
               Fast_evaluate_moments=None, verbose=True):
        """
        Static method: projects the given model
        :param incl: inclination angle of the line-of-sight
        :param b:
        :param Rmax: maximum extent of radial grid
        :param nx: number of grid points
        :param npsi: number of angular subdivisions (for projection)
        :param Fast_evaluate_moments: function to eval. moments
        :param verbose: if True, prints Progressbar
        :return: three maps of density, velocity and velocity dispersion along line-of-sight
        """

        # check if function is callable
        if Fast_evaluate_moments is not None:  # pragma: no cover
            assert hasattr(Fast_evaluate_moments, '__call__')

        # set numpy errors: Ignore invalid values, so that no Warning is raised by sqrt(I2/I1)
        seterr(invalid='ignore')

        # Cython-ing code
        cdef size_t i, j, k
        cdef double ymax, zmax, psi_max, to_radians, sin_incl, cos_incl, dpsi
        cdef double yp, zp, rp, a, I1, I2, I3, psi, ch, sh, xp, x, y, z, R,\
                    phi, abs_z, cphi, sphi
        cdef double dens = 0., Vrot = 0., SigR = 0., Sigp = 0., Sigz = 0.

        cdef ndarray[double_t, ndim=2] dlos, slos, vlos
        cdef ndarray[double_t, ndim=1] yy

        ymax, zmax, psi_max = Rmax, Rmax, 3.
        to_radians = pi / 180.
        sin_incl, cos_incl = sin(incl * to_radians), cos(incl * to_radians)

        dpsi = 2 * psi_max / float(npsi - 1)
        # lam_top, lam_bot = 0., 0.
        dlos, slos, vlos = zeros((2 * nx, 2 * nx)), zeros((2 * nx, 2 * nx)), zeros((2 * nx, 2 * nx))

        # initialize Progressbar if verbose==True
        pbar = None
        if verbose:  # pragma: no cover
            wdgt = ['Projecting: ', widgets.Percentage(), ' ', widgets.Bar(marker=widgets.AnimatedMarker()),
                    ' ', widgets.ETA()]
            pbar = ProgressBar(maxval=2 * nx * nx * npsi, widgets=wdgt).start()

        if scale is 'log':
            yy = logspace(-1., log10(Rmax), num=nx)
            yy = concatenate((-yy[::-1], yy))

        for i in range(nx):
            if scale is 'log':
                yp = yy[nx + i]
            elif scale is 'linear':
                yp = float(i + .5) / (float((nx - 1) + .5)) * ymax
            else:
                raise ValueError("ERROR: Parameter 'scale' must be equal to 'linear' or 'log'!")

            for j in range(2 * nx):
                if scale is 'log':
                    zp = yy[j]
                elif scale is 'linear':
                    zp = (-1. + 2. * j / float(2 * nx - 1)) * zmax
                else:
                    raise ValueError("ERROR: Parameter 'scale' must be equal to 'linear' or 'log'!")

                rp = sqrt(yp * yp + zp * zp)
                a, I1, I2, I3 = max(sqrt(yp * yp + zp * zp), .1 * b), 0., 0., 0.

                for k in range(npsi):
                    # update ProgressBar
                    if verbose:  # pragma: no cover
                        pbar.update((i * 2 * nx + j) * npsi + k)

                    psi = -psi_max + k * dpsi
                    ch, sh, = cosh(psi), sinh(psi)
                    xp = a * sh
                    x, y, z = xp * sin_incl - zp * cos_incl, yp, zp * sin_incl + xp * cos_incl
                    R, phi, abs_z = sqrt(x * x + y * y), atan2(y, x), fabs(z)
                    cphi, sphi = cos(phi), sin(phi)

                    # dens, Vrot, SigR, Sigp, Sigz = Fast_evaluate_moments(R, abs_z)
                    dens = Fast_evaluate_moments(R, abs_z, dens, Vrot, SigR, Sigp, Sigz)

                    I1 += ch * dens
                    I2 += ch * dens * ((SigR * cphi * sin_incl * SigR * cphi * sin_incl
                                        + (Sigp * Sigp - Vrot * Vrot) * sphi * sin_incl * sphi * sin_incl)
                                       + Sigz * cos_incl * Sigz * cos_incl)   # +2*SigRz*sin_incl*cos_incl
                    I3 += ch * dens * Vrot * sphi * sin_incl

                if fabs(I1) > 0:
                    dlos[nx + i, j], slos[nx + i, j], vlos[nx + i, j] = log10(max(a * dpsi * I1, 1e-10)), \
                    sqrt(I2 / I1), I3 / I1

                # symmetrize arrays (the velocity field gets negative for R<0)
                dlos[nx - 1 - i, j], slos[nx - 1 - i, j], vlos[nx - 1 - i, j] = \
                    dlos[nx + i, j], slos[nx + i, j], -vlos[nx + i, j]

                '''
                if rp < 3:
                    lam_top += abs(I3) * rp
                    lam_bot += rp * I1 * sqrt(I2 / I1 + pow(I3 / I1, 2))
                '''

        if verbose:  # pragma: no cover
            pbar.finish()
        # dm, sm, vm = npmax(dlos), npmax(slos), npmax(vlos)

        return dlos, slos, vlos


cdef class LagrangePolynomials(object):

    cdef public int npoly
    cdef public double [:] ar
    cdef public double [:, :] rhl, vrotl, sigRl, sigpl, sigzl

    def __init__(self, int npoly, double [:] ar, double [:, :] rhl,
                 double [:, :] vrotl, double [:, :] sigRl,
                 double [:, :] sigpl, double [:, :] sigzl):

        self.npoly = npoly
        self.ar = ar
        self.rhl = rhl
        self.vrotl = vrotl
        self.sigRl, self.sigpl, self.sigzl = sigRl, sigpl, sigzl

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double [:] even_Legendre(self, double c):
        """
        Static method: gets the even Legendre polynomials at cos(theta)
        :param c: cos(theta)
        :return: list of npoly Legendre polynomials
        """

        # Cython-ize
        cdef double c2
        cdef int l, l2, np
        cdef double [:] pol

        c2 = c * c
        pol = zeros(self.npoly)

        pol[0] = 1
        if self.npoly < 2:  # pragma: no cover
            return pol

        pol[1] = 1.5 * c2 - .5

        for np in prange(2, self.npoly, nogil=True):
            l = 2 * (np - 1)
            l2 = 2 * l
            pol[np] = -pol[np - 2] * l * (l - 1) / float((l2 + 1) * (l2 - 1)) + \
                pol[np - 1] * (c2 - (l2 * l + l2 - 1) / float((l2 - 1) * (l2 + 3)))
            pol[np] *= (l2 + 1) * (l2 + 3) / float((l + 1) * (l + 2))

        return pol

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double fast_evaluate_moments(self, double R, double z,
                                     double Rho, double Vrot, double SigR, double Sigp, double Sigz):

        # Cython-ize
        cdef size_t i, npoly = self.npoly
        cdef double r, c, f
        cdef double [:] pol
        cdef double [:] rhop, vrotp, sigRp, sigpp, sigzp
        # cdef double Rho, Vrot, SigR, Sigp, Sigz

        r = sqrt(R * R + z * z)
        c = z / r

        pol = self.even_Legendre(c)
        rhop, vrotp, sigRp, sigpp, sigzp = self._fast_interpolate_moments(r)

        Rho, Vrot, SigR, Sigp, Sigz = .5 * rhop[0], .5 * vrotp[0], .5 * sigRp[0], .5 * sigpp[0], .5 * sigzp[0]

        for i in prange(1, npoly, nogil=True):
            f = .5 * (4 * i + 1)
            Rho += f * rhop[i] * pol[i]
            Vrot += f * vrotp[i] * pol[i]
            SigR += f * sigRp[i] * pol[i]
            Sigp += f * sigpp[i] * pol[i]
            Sigz += f * sigzp[i] * pol[i]

        return Rho

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline _fast_interpolate_moments(self, double r):
        """
        Private method: assumes r is scalar
        :param r:
        :return:
        """

        #Cython-ize
        cdef size_t i, npoly = self.npoly
        cdef int top, bot
        cdef double f
        cdef double [:] rhop, vrotp, sigRp, sigpp, sigzp

        rhop = zeros(self.npoly)
        vrotp = zeros(self.npoly)
        sigRp = zeros(self.npoly)
        sigpp = zeros(self.npoly)
        sigzp = zeros(self.npoly)

        if r > self.ar[len(self.ar)-1]:
            pass
        else:
            bot = searchsorted(self.ar, r, side='left') - 1
            top = bot + 1

            f = (r - self.ar[bot]) / (self.ar[top] - self.ar[bot])
            for i in prange(npoly, nogil=True):
                rhop[i] = f * self.rhl[top][i] + (1. - f) * self.rhl[bot][i]
                vrotp[i] = f * self.vrotl[top][i] + (1. - f) * self.vrotl[bot][i]
                sigRp[i] = f * self.sigRl[top][i] + (1. - f) * self.sigRl[bot][i]
                sigpp[i] = f * self.sigpl[top][i] + (1. - f) * self.sigpl[bot][i]
                sigzp[i] = f * self.sigzl[top][i] + (1. - f) * self.sigzl[bot][i]

        return rhop, vrotp, sigRp, sigpp, sigzp