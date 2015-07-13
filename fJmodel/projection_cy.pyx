# cython: cdivision=True
# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=False
__author__ = 'lposti'


from numpy import pi, zeros, logspace, concatenate, seterr, searchsorted
from numpy cimport ndarray, double_t
from progressbar import ProgressBar, widgets
from cython.parallel import prange
from cpython cimport array


# C definitions
cdef extern from "math.h":
    double log10 ( double x ) nogil
    double fabs ( double x ) nogil
    double cos ( double x ) nogil
    double sin ( double x ) nogil
    double sqrt ( double x ) nogil
    double cosh ( double x ) nogil
    double sinh ( double x ) nogil
    double atan2 ( double x, double y ) nogil

cpdef projection(double incl, double b, double Rmax, int nx, int npsi,
                 LagrangePolynomials lp, scale='linear', verbose=True):
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

        # set numpy errors: Ignore invalid values, so that no Warning is raised by sqrt(I2/I1)
        seterr(invalid='ignore')

        dlos, slos, vlos = zeros(2 * nx * 2 * nx), zeros(2 * nx * 2 * nx), zeros(2 * nx * 2 * nx)
        cdef array.array dlos_arr = array.array('d', dlos)
        cdef array.array slos_arr = array.array('d', slos)
        cdef array.array vlos_arr = array.array('d', vlos)

        if scale is 'linear':
            compute_projection(incl, b, Rmax, nx, npsi, lp, 0,
                               dlos_arr.data.as_doubles, vlos_arr.data.as_doubles, slos_arr.data.as_doubles)
        elif scale is 'log':
            compute_projection(incl, b, Rmax, nx, npsi, lp, 1,
                               dlos_arr.data.as_doubles, vlos_arr.data.as_doubles, slos_arr.data.as_doubles)
        else:
            raise ValueError("ERROR: Parameter 'scale' must be equal to 'linear' or 'log'!")

        cdef int i, j

        dlos, slos, vlos = zeros((2 * nx, 2 * nx)), zeros((2 * nx, 2 * nx)), zeros((2 * nx, 2 * nx))
        for i in range(2 * nx):
            for j in range(2 * nx):
                dlos[i, j] = dlos_arr[i * 2 * nx + j]
                slos[i, j] = slos_arr[i * 2 * nx + j]
                vlos[i, j] = vlos_arr[i * 2 * nx + j]

        return dlos, slos, vlos


cdef void compute_projection(double incl, double b, double Rmax, int nx, int npsi,
                             LagrangePolynomials lp, int scale,
                             double* dlos, double* vlos, double* slos):

        # Cython-ing code
        cdef int i, j, k

        cdef ndarray[double_t, ndim=1] yy

        cdef double ymax = Rmax, zmax = Rmax, psi_max = 3.
        cdef double to_radians = pi / 180.
        cdef double sin_incl = sin(incl * to_radians), cos_incl = cos(incl * to_radians)

        cdef double dpsi = 2 * psi_max / float(npsi - 1)
        # lam_top, lam_bot = 0., 0.

        # initialize Progressbar if verbose==True
        pbar = None

        wdgt = ['Projecting: ', widgets.Percentage(), ' ', widgets.Bar(marker=widgets.AnimatedMarker()),
                ' ', widgets.ETA()]
        pbar = ProgressBar(maxval=2 * nx * nx * npsi, widgets=wdgt).start()

        if scale == 1:
            yy = logspace(-1., log10(Rmax), num=nx)
            yy = concatenate((-yy[::-1], yy))

        cdef:
            double yp=0., zp =0.
            double rp=0., a=0., I1=0., I2=0., I3=0.
            double psi=0., ch=0., sh=0., xp=0., x=0., z=0., R=0., phi=0., abs_z=0.
            double cphi=0., sphi=0.
            double dens = 0., Vrot = 0., SigR = 0., Sigp = 0., Sigz = 0.
            double density=0.


        for i in prange(nx, nogil=True):

            with gil:
                if scale == 1:
                    yp = yy[nx + i]
                elif scale == 0:
                    yp = float(i + .5) / (float((nx - 1) + .5)) * ymax

            for j in prange(2 * nx):

                with gil:
                    if scale == 1:
                        zp = yy[j]
                    elif scale == 0:
                        zp = (-1. + 2. * j / float(2 * nx - 1)) * zmax

                rp = sqrt(yp * yp + zp * zp)
                a = 0.
                I1 = 0.
                I2 = 0.
                I3 = 0.

                with gil:
                    a = max(rp, .1 * b)

                for k in prange(npsi):
                    with gil:
                        # update ProgressBar
                        pbar.update((i * 2 * nx + j) * npsi + k)

                    psi = -psi_max + k * dpsi
                    ch = cosh(psi)
                    sh = sinh(psi)
                    xp = a * sh
                    x = xp * sin_incl - zp * cos_incl
                    z = zp * sin_incl + xp * cos_incl
                    R = sqrt(x * x + yp * yp)
                    phi = atan2(yp, x)
                    abs_z = fabs(z)
                    cphi = cos(phi)
                    sphi = sin(phi)

                    # dens, Vrot, SigR, Sigp, Sigz = Fast_evaluate_moments(R, abs_z)
                    dens = 0.
                    Vrot = 0.
                    SigR = 0.
                    Sigp = 0.
                    Sigz = 0.
                    lp.fast_evaluate_moments(R, abs_z, &dens, &Vrot, &SigR, &Sigp, &Sigz)

                    I1 += ch * dens
                    I2 += ch * dens * ((SigR * cphi * sin_incl * SigR * cphi * sin_incl
                                        + (Sigp * Sigp - Vrot * Vrot) * sphi * sin_incl * sphi * sin_incl)
                                       + Sigz * cos_incl * Sigz * cos_incl)   # +2*SigRz*sin_incl*cos_incl
                    I3 += ch * dens * Vrot * sphi * sin_incl

                if fabs(I1) > 0:
                    with gil:
                        density = max(a * dpsi * I1, 1e-10)
                    dlos[(nx + i) * 2 * nx + j], slos[(nx + i) * 2 * nx + j], vlos[(nx + i) * 2 * nx + j] = \
                        log10(density), sqrt(I2 / I1), I3 / I1

                # symmetrize arrays (the velocity field gets negative for R<0)
                dlos[(nx - 1 - i) * 2 * nx + j], slos[(nx - 1 - i) * 2 * nx + j], vlos[(nx - 1 - i) * 2 * nx + j] = \
                    dlos[(nx + i) * 2 * nx + j], slos[(nx + i) * 2 * nx + j], -vlos[(nx + i) * 2 * nx + j]

                '''
                if rp < 3:
                    lam_top += abs(I3) * rp
                    lam_bot += rp * I1 * sqrt(I2 / I1 + pow(I3 / I1, 2))
                '''

        pbar.finish()
        # dm, sm, vm = npmax(dlos), npmax(slos), npmax(vlos)


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

    cdef inline void even_Legendre(self, double c, double [:] pol) nogil:
        """
        Static method: gets the even Legendre polynomials at cos(theta)
        :param c: cos(theta)
        :return: list of npoly Legendre polynomials
        """

        # Cython-ize
        cdef double c2
        cdef int l, l2, np

        c2 = c * c

        pol[0] = 1
        if self.npoly < 2:  # pragma: no cover
            return

        pol[1] = 1.5 * c2 - .5

        for np in prange(2, self.npoly):
            l = 2 * (np - 1)
            l2 = 2 * l
            pol[np] = -pol[np - 2] * l * (l - 1) / float((l2 + 1) * (l2 - 1)) + \
                pol[np - 1] * (c2 - (l2 * l + l2 - 1) / float((l2 - 1) * (l2 + 3)))
            pol[np] *= (l2 + 1) * (l2 + 3) / float((l + 1) * (l + 2))

    cdef void fast_evaluate_moments(self, double R, double z,
                                      double* Rho, double* Vrot, double* SigR, double* Sigp, double* Sigz) nogil:

        # Cython-ize
        cdef int i, npoly = self.npoly
        cdef double r, c, f
        cdef double [:] pol
        cdef double [:] rhop, vrotp, sigRp, sigpp, sigzp

        r = sqrt(R * R + z * z)
        c = z / r

        with gil:
            pol = zeros(npoly)
            rhop = zeros(npoly)
            vrotp = zeros(npoly)
            sigRp = zeros(npoly)
            sigpp = zeros(npoly)
            sigzp = zeros(npoly)

        self.even_Legendre(c, pol)
        self._fast_interpolate_moments(r, rhop, vrotp, sigRp, sigpp, sigzp)

        Rho[0] = .5 * rhop[0]
        Vrot[0] = .5 * vrotp[0]
        SigR[0] = .5 * sigRp[0]
        Sigp[0] = .5 * sigpp[0]
        Sigz[0] = .5 * sigzp[0]

        for i in prange(1, npoly):
            f = .5 * (4 * i + 1)
            Rho[0] += f * rhop[i] * pol[i]
            Vrot[0] += f * vrotp[i] * pol[i]
            SigR[0] += f * sigRp[i] * pol[i]
            Sigp[0] += f * sigpp[i] * pol[i]
            Sigz[0] += f * sigzp[i] * pol[i]

    cdef inline void _fast_interpolate_moments(self, double r, double [:] rhop, double [:] vrotp,
                                               double [:] sigRp, double [:] sigpp, double [:] sigzp) nogil:
        """
        Private method: assumes r is scalar
        :param r:
        :return:
        """

        #Cython-ize
        cdef int i, npoly = self.npoly
        cdef int top, bot, ar_size
        cdef double f

        with gil:
            ar_size = len(self.ar)

        if r > self.ar[ar_size-1]:
            pass
        else:
            with gil:
                bot = searchsorted(self.ar, r, side='left') - 1
            top = bot + 1

            f = (r - self.ar[bot]) / (self.ar[top] - self.ar[bot])
            for i in prange(0, npoly):
                rhop[i] = f * self.rhl[top][i] + (1. - f) * self.rhl[bot][i]
                vrotp[i] = f * self.vrotl[top][i] + (1. - f) * self.vrotl[bot][i]
                sigRp[i] = f * self.sigRl[top][i] + (1. - f) * self.sigRl[bot][i]
                sigpp[i] = f * self.sigpl[top][i] + (1. - f) * self.sigpl[bot][i]
                sigzp[i] = f * self.sigzl[top][i] + (1. - f) * self.sigzl[bot][i]