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
from voronoi import voronoi_2d_binning
from math import sqrt as msqrt
from numpy import fromstring, zeros, searchsorted, sqrt, asarray, ndarray, cos, sin, pi, arccos, trapz,\
    cosh, sinh, arctan2, power, log10, linspace, seterr, inf, meshgrid, reshape, isnan, abs, ogrid, logspace,\
    concatenate, array
from progressbar import ProgressBar, widgets
from scipy.integrate import tplquad
from scipy.optimize import brentq
from scipy.ndimage.filters import gaussian_filter


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

            #
            # get header (was recently modified)
            #
            self.nr, self.npoly, self.ngauss = fromstring(getline(self.fname, self._line), dtype=int, sep=' ')
            self._line += 1

            self.dphi_h, self.dz_h, self.dphi_g, self.dz_g =\
                fromstring(getline(self.fname, self._line), dtype=float, sep=' ')
            self._line += 1

            self.chi, self.M0, self.r0 = fromstring(getline(self.fname, self._line), dtype=float, sep=' ')
            self._line += 1

            self.alpha, self.beta = fromstring(getline(self.fname, self._line), dtype=float, sep=' ')
            self._line += 1
            # header ended
            #####################################

            #
            # start reading the input file
            #
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
            # end of file
            #####################################

            # initialize to None
            self.dlos, self.slos, self.vlos = None, None, None
            self.xmap, self.ymap = None, None
            self.r_eff, self.v_scale = None, None

            # initialize to None
            self.binNum, self.xNode, self.yNode, self.xBar, self.yBar, self.sn, self.nPixels, self.scale = \
                None, None, None, None, None, None, None, None

            # compute spherically averaged Mass
            self.mass = 4. * pi * trapz(self.ar * self.ar * self.rho(self.ar, 0), self.ar)

            # compute intrinsic ellipticity
            try:
                self.eps = self._get_ellipticity()
            except ValueError:
                self.eps = zeros(self.nr)
                print " -- WARNING: ellipticity not properly computed!!"

        except AssertionError:  # pragma: no cover
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

        # check if R<0: in that case we need to symmetrize the field

        if R.size > 1:
            index = searchsorted(R, 0.)
            if index > 0:
                vrot_map = self._getq(R, z, self.vrotl)
                vrot_map[0:index, :] = -vrot_map[0:index, :]
                return vrot_map
            else:
                return self._getq(R, z, self.vrotl)
        else:
            if R > 0:
                return self._getq(R, z, self.vrotl)
            else:
                return -self._getq(R, z, self.vrotl)

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

    def virial(self, verbose=True, ret=False):
        """
        Computes and prints the quantity involved in the tensor virial theorem.
        We judge a system to be relaxed to self-consistency if the ratios of the potential over
        kinetic energies (diagonal and off diagonal tensor terms) are reasonably close to 2
        :param verbose: bool, if True prints the virial statistic
        :param ret: bool, if True returns the diagonal terms of the virial tensor
        :return: Prints the statistic for the tensor virial theorem. If ret=True, returns the diagonal
                 terms of the virial tensor as floats
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

        if verbose:  # pragma: no cover
            print "  Virial statistic: "
            print "  Mass(%3.1f): %f %f" % (self.ar[-1], pow(self.ar[-1], 2) * self.Pr[-1, 0], self.mass)
            print "  KE, PE, W/K = %f %f %f " % (KRR + Kzz, pot, pot / (KRR + Kzz))
            print "  Kxx, Wxx, Wxx/Kxx = %f %f %f" % (KRR, WRR, WRR / KRR)
            print "  Kzz, Wzz, Wzz/Kzz = %f %f %f" % (Kzz, Wzz, Wzz / Kzz)

        if ret:
            return pot / (KRR + Kzz), WRR / KRR, Wzz / Kzz

    def compare_mass(self, verbose=True):

        massfJ = self.computeMassIntegral(alpha=self.alpha, beta=self.beta, M0=self.M0, r0=self.r0,
                                          dphih=self.dphi_h, dzh=self.dz_h, dphig=self.dphi_g, dzg=self.dz_g)

        if verbose:  # pragma: no cover
            print "Mass: f(J) model = %f, eq. (37)=%f, ratio=%f" % (self.mass, massfJ, self.mass / massfJ)
        return self.mass / massfJ

    @staticmethod
    def computeMassIntegral(alpha, beta, M0, r0, dphih, dzh, dphig, dzg):

        J0 = msqrt(M0 * r0)
        h = lambda Jr, Jphi, Jz: Jr + dphih * abs(Jphi) + dzh * Jz
        g = lambda Jr, Jphi, Jz: Jr + dphig * abs(Jphi) + dzg * Jz
        DF = lambda Jr, Jphi, Jz: M0 / pow(J0, 3) * pow(1. + J0 / h(Jr, Jphi, Jz), alpha) /\
            pow(1 + g(Jr, Jphi, Jz) / J0, beta)

        if beta > 3:
            # integral is doubled since -inf< Jphi <inf
            massfJ = 2. * tplquad(DF, 0, inf, lambda x: 0, lambda x: inf,
                                  lambda x, y: 0, lambda x, y: inf)[0]

        else:  # pragma: no cover
            raise NotImplementedError("The mass integral diverges and can't be computed in action space.")

        return massfJ

    def project(self, inclination, nx=80, npsi=31, b=1., scale='linear', Rmax=None, verbose=True):
        """
        Project the f(J) model along a line-of-sight specified by the inclination (in degrees)
        :param inclination: inclination of the line-of-sight desired for the projection (in degrees, 90 is edge-on)
        :param nx: number of grid points
        :param npsi: number of angular subdivisions (for projection)
        :param b:
        :param Rmax: maximum extent of radial grid
        :param verbose: if True, prints progressbar
        :return: x, y (ndarray) [-Rmax,Rmax] to be passed to contour
        """

        if Rmax is None:  # pragma: no cover
            Rmax = self.ar[-1]

        # nx, npsi = 60, 81
        self.dlos, self.slos, self.vlos = self.projection(incl=inclination, b=b, Rmax=Rmax, nx=nx,
                                                          npsi=npsi, scale=scale,
                                                          Fast_evaluate_moments=self._fast_evaluate_moments,
                                                          verbose=verbose)

        # remove nans in the maps
        for j in range(2 * nx):
            for k in range(2 * nx):
                if isnan(self.slos[j, k]):
                    self.slos[j, k] = 1e-10
                if isnan(self.vlos[j, k]):
                    self.vlos[j, k] = 1e-10

        if scale is 'log':
            self.xmap = logspace(-1., log10(Rmax), num=nx)
            self.xmap = concatenate((-self.xmap[::-1], self.xmap))

            self.ymap = logspace(-1., log10(Rmax), num=nx)
            self.ymap = concatenate((-self.ymap[::-1], self.ymap))

        elif scale is 'linear':
            self.xmap = linspace(-Rmax, Rmax, num=2 * nx)
            self.ymap = linspace(-Rmax, Rmax, num=2 * nx)
        else:
            raise ValueError("ERROR: Parameter 'scale' must be equal to 'linear' or 'log'!")

        #
        # find the projected half mass radius
        #

        # grid density (exclude minimum at -10)
        dd = linspace((self.dlos[self.dlos > -9]).min(), self.dlos.max(), num=nx)

        # compute projected mass array
        mm = []
        for i in range(len(dd)):
            w = self.dlos > dd[i]
            mm.append(pow(Rmax / nx, 2) * power(10., self.dlos[w]).sum())

        # compute semi-major axis at each density level
        RR = []
        for i in range(len(mm)):
            idx = abs(self.dlos[:, len(self.dlos) / 2] - dd[i]).argmin()
            RR.append(abs(self.xmap[idx]))

        # minimize to get the effective radius
        id_r_eff = abs(mm - mm[0] / 2).argmin()
        self.r_eff = RR[id_r_eff]
        print "Effective radius:", self.r_eff, "[M/2-M(Re)]/M=", abs(mm - mm[0] / 2).min() / mm[0]

        #
        # compute scale velocity
        #

        w = self.dlos > dd[id_r_eff]

        self.v_scale = (power(10., self.dlos[w]) * sqrt(self.vlos[w] ** 2 + self.slos[w] ** 2)).sum() /\
            power(10., self.dlos[w]).sum()

        print "Velocity scale:", self.v_scale

        return self.xmap, self.ymap

    @staticmethod
    def projection(incl, b, Rmax, nx, npsi, scale='linear', Fast_evaluate_moments=None, rho=None, vrot=None,
                   sigR=None, sigp=None, sigz=None, verbose=True):
        """
        Static method: projects the given model
        :param incl: inclination angle of the line-of-sight
        :param b:
        :param Rmax: maximum extent of radial grid
        :param nx: number of grid points
        :param npsi: number of angular subdivisions (for projection)
        :param Fast_evaluate_moments: function to eval. moments
        :param rho: rho function
        :param vrot: vrot function
        :param sigR: sigR function
        :param sigp: sigp function
        :param sigz: sigz function
        :param verbose: if True, prints Progressbar
        :return: three maps of density, velocity and velocity dispersion along line-of-sight
        """

        # check if functions are callable
        if Fast_evaluate_moments is not None:  # pragma: no cover
            assert hasattr(Fast_evaluate_moments, '__call__')
        if rho is not None:  # pragma: no cover
            assert hasattr(rho, '__call__')
        if vrot is not None:  # pragma: no cover
            assert hasattr(vrot, '__call__')
        if sigR is not None:  # pragma: no cover
            assert hasattr(sigR, '__call__')
        if sigp is not None:  # pragma: no cover
            assert hasattr(sigp, '__call__')
        if sigz is not None:  # pragma: no cover
            assert hasattr(sigz, '__call__')

        # set numpy errors: Ignore invalid values, so that no Warning is raised by sqrt(I2/I1)
        seterr(invalid='ignore')

        ymax, zmax, psi_max = Rmax, Rmax, 3.
        to_radians = pi / 180.
        sin_incl, cos_incl = sin(incl * to_radians), cos(incl * to_radians)

        dpsi = 2 * psi_max / float(npsi - 1)
        lam_top, lam_bot = 0., 0.
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
                    R, phi, abs_z = sqrt(x * x + y * y), arctan2(y, x), abs(z)
                    cphi, sphi = cos(phi), sin(phi)

                    if Fast_evaluate_moments is not None:
                        dens, Vrot, SigR, Sigp, Sigz = Fast_evaluate_moments(R, abs_z)
                    else:  # pragma: no cover
                        dens, Vrot = rho(R, abs_z), vrot(R, abs_z)
                        SigR, Sigp, Sigz = sigR(R, abs_z), sigp(R, abs_z), sigz(R, abs_z)

                    I1 += ch * dens
                    I2 += ch * dens * ((power(SigR * cphi * sin_incl, 2)
                                        + (Sigp * Sigp - Vrot * Vrot) * power(sphi * sin_incl, 2))
                                       + power(Sigz * cos_incl, 2))   # +2*SigRz*sin_incl*cos_incl
                    I3 += ch * dens * Vrot * sphi * sin_incl

                dlos[nx + i, j], slos[nx + i, j], vlos[nx + i, j] = log10(max(a * dpsi * I1, 1e-10)), \
                    sqrt(I2 / I1), I3 / I1

                # symmetrize arrays (the velocity field gets negative for R<0)
                dlos[nx - 1 - i, j], slos[nx - 1 - i, j], vlos[nx - 1 - i, j] = \
                    dlos[nx + i, j], slos[nx + i, j], -vlos[nx + i, j]
                if rp < 3:
                    lam_top += abs(I3) * rp
                    lam_bot += rp * I1 * sqrt(I2 / I1 + pow(I3 / I1, 2))

        if verbose:  # pragma: no cover
            pbar.finish()
        # dm, sm, vm = npmax(dlos), npmax(slos), npmax(vlos)

        return dlos, slos, vlos

    def light_profile(self, inclination=90, nx=80, npsi=31, scale='log', Re_model=1., Re_data=1., **kwargs):

        x, y = self.project(inclination=inclination, nx=nx, npsi=npsi, scale=scale, **kwargs)

        R_arcsec = linspace(0.2, 250., num=200)
        R = R_arcsec * Re_model / Re_data

        d_psf = gaussian_filter(self.dlos, 1., mode='nearest')

        """
        THIS IS THE RADIAL SURFACE BRIGHTNESS PROFILE
        s = []
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] ** 2 + y[j] ** 2 <= R[0]:
                    s.append(10. ** self.dlos[i, j])
        light_profile = [array(s).mean()]

        for k in range(1, len(R)):
            s = []
            for i in range(len(x)):
                for j in range(len(y)):
                    if (x[i] ** 2 + y[j] ** 2 <= R[k]) &\
                            (x[i] ** 2 + y[j] ** 2 > R[k - 1]):
                        s.append(10. ** self.dlos[i, j])
            light_profile.append(array(s).mean())
        """

        # if computations larger than 30x30x100 initialize pbar
        pbar = None
        if nx * nx * len(R) > 9e4:  # pragma: no cover
            wdgt = [widgets.FormatLabel('Computing growth curve: '), widgets.Percentage()]
            pbar = ProgressBar(maxval=nx * nx * len(R), widgets=wdgt).start()

        # compute the growth curve for the model (i.e. integrated flux within circular apertures)

        growth_curve = []
        for k in range(0, len(R)):
            s = []
            for i in range(len(x)):
                for j in range(len(y)):

                    # update ProgressBar
                    if nx * nx * len(R) > 9e4:  # pragma: no cover
                        pbar.update((k * (nx - 1) + i) * (nx - 1) + j)

                    if x[i] ** 2 + y[j] ** 2 <= R[k] ** 2:
                        # s.append(10. ** self.dlos[i, j])
                        s.append(10. ** d_psf[i, j])
            growth_curve.append(array(s).sum())

        if nx * nx * len(R) > 9e4:  # pragma: no cover
            pbar.finish()

        return R_arcsec, -2.5 * log10(growth_curve)

    def voronoiBin(self, **kwargs):

        if self.dlos is None or self.slos is None or self.vlos is None:  # pragma: no cover
            if kwargs == {}:
                x, y = self.project(90, nx=30, npsi=31)
            else:
                x, y = self.project(**kwargs)
        else:
            x, y = self.xmap, self.ymap

        Y, X = meshgrid(x, y)

        if self.dlos is not None:  # pragma: no cover
            density = self.dlos
        else:
            raise TypeError(" -- Critical ERROR: dlos is still None Type!")

        # reshape arrays for voronoi binning
        X1 = X.reshape(len(x) * len(x))
        Y1 = Y.reshape(len(y) * len(y))
        D = power(10., reshape(density, len(x) * len(x)))

        # create noise proportional to sqrt(signal)
        N = sqrt(D)

        # apply Cappellari's Voronoi binning procedure
        self.binNum, self.xNode, self.yNode, self.xBar, self.yBar, self.sn, self.nPixels, self.scale =\
            voronoi_2d_binning(X1, Y1, D, N, targetSN=0.01)

        return X1, Y1

    def _get_ellipticity(self):
        """
        Private method: gets the ellipticity profile along the radial grid.
        The ellipticity is defined as e=1-z/R, where z is the position along the minor axis where the
        (intrinsic) density is equal to that along the major axis at position R.
        :return: ellipticity profile (array type)
        """

        eps = zeros(self.nr)
        minR, maxR = self.ar[0] / 2, self.ar[-1] * 2

        for i in range(self.nr):
            f = lambda z: self.rho(self.ar[i], 0.) - self.rho(0., z)
            # ellipticity defined as eps = 1-z/R
            eps[i] = 1. - brentq(f, minR, maxR) / self.ar[i]

        return eps

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
        except AssertionError:  # pragma: no cover
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
        if npoly < 2:  # pragma: no cover
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

    def _fast_evaluate_moments(self, R, z):

        assert R.size is 1
        assert z.size is 1

        r = sqrt(R * R + z * z)
        c = z / r

        pol = self._evenlegend(c)
        rhop, vrotp, sigRp, sigpp, sigzp = self._fast_interpolate_moments(r)

        Rho, Vrot, SigR, Sigp, Sigz = .5 * rhop[0], .5 * vrotp[0], .5 * sigRp[0], .5 * sigpp[0], .5 * sigzp[0]
        for i in range(1, self.npoly):
            f = .5 * (4 * i + 1)
            Rho += f * rhop[i] * pol[i]
            Vrot += f * vrotp[i] * pol[i]
            SigR += f * sigRp[i] * pol[i]
            Sigp += f * sigpp[i] * pol[i]
            Sigz += f * sigzp[i] * pol[i]

        return Rho, Vrot, SigR, Sigp, Sigz

    def _fast_interpolate_moments(self, r):
        """
        Private method: assumes r is scalar
        :param r:
        :return:
        """

        # check r is a scalar
        assert r.size is 1

        rhop = zeros(self.npoly, dtype=float)
        vrotp = zeros(self.npoly, dtype=float)
        sigRp = zeros(self.npoly, dtype=float)
        sigpp = zeros(self.npoly, dtype=float)
        sigzp = zeros(self.npoly, dtype=float)

        if r > self.ar[-1]:
            pass
        else:
            bot = searchsorted(self.ar, r, side='left') - 1
            top = bot + 1

            f = (r - self.ar[bot]) / (self.ar[top] - self.ar[bot])
            for i in range(self.npoly):
                rhop[i] = f * self.rhl[top][i] + (1 - f) * self.rhl[bot][i]
                vrotp[i] = f * self.vrotl[top][i] + (1 - f) * self.vrotl[bot][i]
                sigRp[i] = f * self.sigRl[top][i] + (1 - f) * self.sigRl[bot][i]
                sigpp[i] = f * self.sigpl[top][i] + (1 - f) * self.sigpl[bot][i]
                sigzp[i] = f * self.sigzl[top][i] + (1 - f) * self.sigzl[bot][i]

        return rhop, vrotp, sigRp, sigpp, sigzp

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
                if Pr2 is not None and nr is not None:  # pragma: no cover
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
                if Pr2 is not None and nr is not None:  # pragma: no cover
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
                    if Pr2 is not None and nr is not None:  # pragma: no cover
                        d2phip[k] += f2 * (Pr2[thr, k] - Pr2[bot, k] - f3 * (Pr2[top, k] - Pr2[bot, k]))

        if Pr is not None:  # pragma: no cover
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
            ngauss is not None:  # pragma: no cover

                self.phil, self.Pr, self.Pr2 = phil, Pr, Pr2
                self.ar = ar
                self.nr, self.npoly, self.ngauss = nr, npoly, ngauss

        else:  # pragma: no cover
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

        else:  # pragma: no cover
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
            if r < self.ar[0]:  # pragma: no cover
                r = self.ar[0]

            s, c = R / r, z / r
            if c == 1:  # pragma: no cover
                c -= 1e-8

            theta = arccos(c)
            dphidr = self._dtheta(r, c) * (c / r) + self._dr(r, theta) * s

        # R array case
        elif R.size > 1 and z.size == 1:
            dphidr = zeros(R.size)
            for k in range(R.size):
                r = sqrt(z * z + R[k] * R[k])
                if r < self.ar[0]:  # pragma: no cover
                    r = self.ar[0]

                s, c = R[k] / r, z / r
                if c == 1:  # pragma: no cover
                    c -= 1e-8

                theta = arccos(c)
                dphidr[k] = self._dtheta(r, c) * (c / r) + self._dr(r, theta) * s

        # z array case
        elif R.size == 1 and z.size > 1:
            dphidr = zeros(z.size)
            for j in range(z.size):
                r = sqrt(z[j] * z[j] + R * R)
                if r < self.ar[0]:  # pragma: no cover
                    r = self.ar[0]

                s, c = R / r, z[j] / r
                if c == 1:  # pragma: no cover
                    c -= 1e-8

                theta = arccos(c)
                dphidr[j] = self._dtheta(r, c) * (c / r) + self._dr(r, theta) * s

        # [R, z] array case
        elif R.size > 1 and z.size > 1:
            dphidr = zeros((R.size, z.size))
            for k in range(R.size):
                for j in range(z.size):
                    r = sqrt(z[j] * z[j] + R[k] * R[k])
                    if r < self.ar[0]:  # pragma: no cover
                        r = self.ar[0]

                    s, c = R[k] / r, z[j] / r
                    if c == 1:  # pragma: no cover
                        c -= 1e-8

                    theta = arccos(c)
                    dphidr[k, j] = self._dtheta(r, c) * (c / r) + self._dr(r, theta) * s

        else:  # pragma: no cover
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
            if r < self.ar[0]:  # pragma: no cover
                r = self.ar[0]

            s, c = R / r, z / r
            if c == 1:  # pragma: no cover
                c -= 1e-8

            theta = arccos(c)
            dphidz = self._dtheta(r, c) * (-s / r) + self._dr(r, theta) * c

        # R array case
        elif R.size > 1 and z.size == 1:
            dphidz = zeros(R.size)
            for k in range(R.size):
                r = sqrt(z * z + R[k] * R[k])
                if r < self.ar[0]:  # pragma: no cover
                    r = self.ar[0]

                s, c = R[k] / r, z / r
                if c == 1:  # pragma: no cover
                    c -= 1e-8

                theta = arccos(c)
                dphidz[k] = self._dtheta(r, c) * (-s / r) + self._dr(r, theta) * c

        # z array case
        elif R.size == 1 and z.size > 1:
            dphidz = zeros(z.size)
            for j in range(z.size):
                r = sqrt(z[j] * z[j] + R * R)
                if r < self.ar[0]:  # pragma: no cover
                    r = self.ar[0]

                s, c = R / r, z[j] / r
                if c == 1:  # pragma: no cover
                    c -= 1e-8

                theta = arccos(c)
                dphidz[j] = self._dtheta(r, c) * (-s / r) + self._dr(r, theta) * c

        # [R, z] array case
        elif R.size > 1 and z.size > 1:
            dphidz = zeros((R.size, z.size))
            for k in range(R.size):
                for j in range(z.size):
                    r = sqrt(z[j] * z[j] + R[k] * R[k])
                    if r < self.ar[0]:  # pragma: no cover
                        r = self.ar[0]

                    s, c = R[k] / r, z[j] / r
                    if c == 1:  # pragma: no cover
                        c -= 1e-8

                    theta = arccos(c)
                    dphidz[k, j] = self._dtheta(r, c) * (-s / r) + self._dr(r, theta) * c

        else:  # pragma: no cover
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
        if npoly < 2:  # pragma: no cover
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
