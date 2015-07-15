# cython: cdivision=True
# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
__author__ = 'lposti'


from numpy import pi, zeros, logspace, linspace, concatenate, searchsorted
from numpy cimport ndarray, double_t
from progressbar import ProgressBar, widgets
from cython.parallel import prange


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


cpdef projection(double incl, double b, double Rmax, double Rmin, int nx, int npsi,
                 int npoly, int nr, ndarray[double_t, ndim=1] ar,
                 ndarray[double_t, ndim=1] rhl, ndarray[double_t, ndim=1] vrotl, ndarray[double_t, ndim=1] sigRl,
                 ndarray[double_t, ndim=1] sigpl, ndarray[double_t, ndim=1] sigzl,
                 scale='linear', verbose=True):

        # Cython-ize code
        cdef int i, j, k
        cdef ndarray[double_t, ndim=1] yy

        cdef:
            double ymax = Rmax, zmax = Rmax, psi_max = 3.
            double to_radians = pi / 180.
            double sin_incl = sin(incl * to_radians), cos_incl = cos(incl * to_radians)
            double dpsi = 2 * psi_max / float(npsi - 1)

        dlos, slos, vlos = zeros((2 * nx, 2 * nx)), zeros((2 * nx, 2 * nx)), zeros((2 * nx, 2 * nx))

        # initialize Progressbar if verbose==True
        pbar = None
        wdgt = ['Projecting: ', widgets.Percentage(), ' ', widgets.Bar(marker=widgets.AnimatedMarker()),
                ' ', widgets.ETA()]
        pbar = ProgressBar(maxval=2 * nx * nx * npsi, widgets=wdgt).start()

        if scale is 'log':
            yy = logspace(log10(Rmin), log10(Rmax), num=nx)
            yy = concatenate((-yy[::-1], yy))
        elif scale is 'linear':
            yy = linspace(Rmin, Rmax, num=nx)
            yy = concatenate((-yy[::-1], yy))
        else:
            raise ValueError("Parameter 'scale' must be set to 'linear' or 'log'!")

        cdef:
            double yp=0., zp =0.
            double rp=0., a=0., I1=0., I2=0., I3=0.
            double psi=0., ch=0., sh=0., xp=0., x=0., z=0., R=0., phi=0., abs_z=0.
            double cphi=0., sphi=0.
            double dens = 0., Vrot = 0., SigR = 0., Sigp = 0., Sigz = 0.
            double density=0.


        for i in range(nx):

            yp = yy[nx + i]

            for j in range(2 * nx):

                zp = yy[j]

                rp = sqrt(yp * yp + zp * zp)
                a = 0.
                I1 = 0.
                I2 = 0.
                I3 = 0.

                a = max(rp, .1 * b)

                for k in range(npsi):
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

                    dens = 0.
                    Vrot = 0.
                    SigR = 0.
                    Sigp = 0.
                    Sigz = 0.
                    fast_moments(npoly, nr, ar, &rhl[0], &vrotl[0], &sigRl[0], &sigpl[0], &sigzl[0],
                                 R, abs_z, &dens, &Vrot, &SigR, &Sigp, &Sigz)

                    I1 += ch * dens
                    I2 += ch * dens * ((SigR * cphi * sin_incl * SigR * cphi * sin_incl
                                        + (Sigp * Sigp - Vrot * Vrot) * sphi * sin_incl * sphi * sin_incl)
                                       + Sigz * cos_incl * Sigz * cos_incl)   # +2*SigRz*sin_incl*cos_incl
                    I3 += ch * dens * Vrot * sphi * sin_incl


                density = max(a * dpsi * I1, 1e-10)
                dlos[nx + i, j], slos[nx + i, j], vlos[nx + i, j] = \
                    log10(density), sqrt(I2 / I1), I3 / I1

                # symmetrize arrays (the velocity field gets negative for R<0)
                dlos[nx - 1 - i, j], slos[nx - 1 - i, j], vlos[nx - 1 - i, j] = \
                    dlos[nx + i, j], slos[nx + i, j], -vlos[nx + i, j]

                '''
                if rp < 3:
                    lam_top += abs(I3) * rp
                    lam_bot += rp * I1 * sqrt(I2 / I1 + pow(I3 / I1, 2))
                '''

        pbar.finish()
        # dm, sm, vm = npmax(dlos), npmax(slos), npmax(vlos)
        return dlos, slos, vlos


cdef void fast_moments(int npoly, int nr, ndarray[double_t, ndim=1] ar,
                         double* rhl, double* vrotl, double* sigRl, double* sigpl, double* sigzl,
                         double R, double z, double* Rho, double* Vrot, double* SigR, double* Sigp, double* Sigz):

    # Cython-ize
    cdef int i
    cdef double r, c, f, f_tb
    cdef ndarray[double_t, ndim=1] pol
    cdef ndarray[double_t, ndim=1] rhop, vrotp, sigRp, sigpp, sigzp

    pol = zeros(npoly)
    rhop = zeros(npoly)
    vrotp = zeros(npoly)
    sigRp = zeros(npoly)
    sigpp = zeros(npoly)
    sigzp = zeros(npoly)

    '''
    Compute pol
    '''
    r = sqrt(R * R + z * z)
    c = z / r

    cdef double c2
    cdef int l, l2, np
    cdef int top, bot, ar_size

    c2 = c * c

    pol[0] = 1
    pol[1] = 1.5 * c2 - .5

    for np in range(2, npoly):
        l = 2 * (np - 1)
        l2 = 2 * l
        pol[np] = -pol[np - 2] * l * (l - 1) / float((l2 + 1) * (l2 - 1)) + \
                  pol[np - 1] * (c2 - (l2 * l + l2 - 1) / float((l2 - 1) * (l2 + 3)))
        pol[np] *= (l2 + 1) * (l2 + 3) / float((l + 1) * (l + 2))

    if r > ar[nr-1]:
        pass
    else:

        bot = searchsorted(ar, r, side='left') - 1
        top = bot + 1

        f = (r - ar[bot]) / (ar[top] - ar[bot])
        for i in range(0, npoly):
            rhop[i] = f * rhl[top * npoly + i] + (1. - f) * rhl[bot * npoly + i]
            vrotp[i] = f * vrotl[top * npoly + i] + (1. - f) * vrotl[bot * npoly + i]
            sigRp[i] = f * sigRl[top * npoly + i] + (1. - f) * sigRl[bot * npoly + i]
            sigpp[i] = f * sigpl[top * npoly + i] + (1. - f) * sigpl[bot * npoly + i]
            sigzp[i] = f * sigzl[top * npoly + i] + (1. - f) * sigzl[bot * npoly + i]

    Rho[0] = 0.
    Vrot[0] = 0.
    SigR[0] = 0.
    Sigp[0] = 0.
    Sigz[0] = 0.
    for i in range(0, npoly):
        f = .5 * (4 * i + 1)
        Rho[0] += f * rhop[i] * pol[i]
        Vrot[0] += f * vrotp[i] * pol[i]
        SigR[0] += f * sigRp[i] * pol[i]
        Sigp[0] += f * sigpp[i] * pol[i]
        Sigz[0] += f * sigzp[i] * pol[i]
    '''

    bot = searchsorted(ar, r, side='left') - 1
    top = bot + 1
    f_tb = (r - ar[bot]) / (ar[top] - ar[bot])

    Rho[0] = 0.5 * (f_tb * rhl[top * npoly] + (1. - f_tb) * rhl[bot * npoly]) + \
        5. / 2. * (f_tb * rhl[top * npoly + 1] + (1. - f_tb) * rhl[bot * npoly + 1]) * (1.5 * c2 - .5)
    Vrot[0] = 0.5 * (f_tb * vrotl[top * npoly] + (1. - f_tb) * vrotl[bot * npoly])+ \
        5. / 2. * (f_tb * vrotl[top * npoly + 1] + (1. - f_tb) * vrotl[bot * npoly + 1]) * (1.5 * c2 - .5)
    SigR[0] = 0.5 * (f_tb * sigRl[top * npoly] + (1. - f_tb) * sigRl[bot * npoly])+ \
        5. / 2. * (f_tb * sigRl[top * npoly + 1] + (1. - f_tb) * sigRl[bot * npoly + 1]) * (1.5 * c2 - .5)
    Sigp[0] = 0.5 * (f_tb * sigpl[top * npoly] + (1. - f_tb) * sigpl[bot * npoly])+ \
        5. / 2. * (f_tb * sigpl[top * npoly + 1] + (1. - f_tb) * sigpl[bot * npoly + 1]) * (1.5 * c2 - .5)
    Sigz[0] = 0.5 * (f_tb * sigzl[top * npoly] + (1. - f_tb) * sigzl[bot * npoly])+ \
        5. / 2. * (f_tb * sigzl[top * npoly + 1] + (1. - f_tb) * sigzl[bot * npoly + 1]) * (1.5 * c2 - .5)

    for i in prange(2, npoly, schedule='static', nogil=True):
        f = .5 * (4 * i + 1)
        Rho[0] += f * (f_tb * rhl[top * npoly + i] + (1. - f_tb) * rhl[bot * npoly + i]) * pol[i]
        Vrot[0] += f * (f_tb * vrotl[top * npoly + i] + (1. - f_tb) * vrotl[bot * npoly + i]) * pol[i]
        SigR[0] += f * (f_tb * sigRl[top * npoly + i] + (1. - f_tb) * sigRl[bot * npoly + i]) * pol[i]
        Sigp[0] += f * (f_tb * sigpl[top * npoly + i] + (1. - f_tb) * sigpl[bot * npoly + i]) * pol[i]
        Sigz[0] += f * (f_tb * sigzl[top * npoly + i] + (1. - f_tb) * sigzl[bot * npoly + i]) * pol[i]
    '''
