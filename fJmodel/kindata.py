# -*- coding: utf-8 -*-
__author__ = 'lposti'

from fJmodel import FJmodel
from sauron import sauron
from numpy import genfromtxt, zeros, linspace, exp, log10, column_stack, round, empty, nan, rot90, \
    meshgrid, radians, ones_like, where, average, sqrt, gradient, power, dstack, array, full, around, \
    asarray
from numpy.ma import masked_array
from linecache import getline
from math import sin, cos, tan, pi
from numpy import sin as numpy_sin
from numpy import cos as numpy_cos
from numpy import max as npmax
from numpy import min as npmin
from numpy import abs as npabs
from numpy import reshape as numpy_reshape
from scipy.spatial import distance
from scipy.interpolate import interp2d
from scipy.optimize import minimize, curve_fit
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pylab as plt
import pyfits


CALIFA_arcsec_spaxel = 1.


class KinData(object):

    def __init__(self, directory=None, mge_file=None, kin_data_file=None, aperture_file=None, bins_file=None,
                 fits_file=None, gc_file=None):

        if directory is not None:
            self.mge_file = directory + "/MGE_rband.txt"
            self.kin_data_file = directory + "/kin_data.dat"
            self.aperture_file = directory + "/aperture.dat"
            self.bins_file = directory + "/bins.dat"

            if directory[-1] == '/':
                self.gal_name = directory[-8:-1]
                self.conf_file = directory[:-8] + 'fits_rband/CALIFA_ETGs.conf'
            else:
                self.gal_name = directory[-7:]
                self.conf_file = directory[:-7] + 'fits_rband/CALIFA_ETGs.conf'
            print "Galaxy:", self.gal_name

            self.fits_file = directory + '/' + self.gal_name + '.V1200.rscube_INDOUSv2_SN20_stellar_kin.fits'
            self.gc_file = directory + '/' + self.gal_name + '_GCs.dat'

        elif mge_file is not None and kin_data_file is not None and \
                aperture_file is not None and bins_file is not None and \
                fits_file is not None:
            self.mge_file = mge_file
            self.kin_data_file = kin_data_file
            self.aperture_file = aperture_file
            self.bins_file = bins_file
            self.fits_file = fits_file
            self.gc_file = gc_file

        else:
            raise ValueError("ERROR: either pass the directory where the MGE, kinematic"
                             "and aperture data are or pass the filenames directly.")

        self.Re = self._get_effective_radius()
        self.angle = float(getline(self.aperture_file, 4).split()[0])
        self.R_arcsec, self.gc = None, None

        try:
            self.R_arcsec, self.gc = self._read_growth_curve_file()
        except IOError:
            print "  - WARNING: Could not find growth curve file."

    def _get_mge(self, xt=None, yt=None, angle=0., size=100):

        dat = genfromtxt(self.mge_file)
        sb = dat[:, 0]
        sigma = dat[:, 1]
        q = dat[:, 2]

        # set the size of the map to the maximum sigma of the MGE expansion
        px_size = npmax(sigma)

        if xt is None and yt is None and size is not None:
            x = linspace(-px_size, px_size, num=size)
            y = linspace(-px_size, px_size, num=size)
            nx, ny = size, size
        else:
            x, y = xt, yt
            nx, ny = len(xt), len(yt)

        image = zeros((nx, ny))

        for i in range(0, nx):
            for j in range(0, ny):
                x_rotated = cos(radians(angle)) * x[i] - sin(radians(angle)) * y[j]
                y_rotated = sin(radians(angle)) * x[i] + cos(radians(angle)) * y[j]
                for k in range(0, len(sb)):
                    image[i, j] += sb[k] * exp(-(pow(x_rotated, 2) + pow(y_rotated, 2) / pow(q[k], 2))
                                               / (2. * pow(sigma[k], 2)))

        image /= npmax(image)

        if xt is None and yt is None:
            return image, x, y
        else:
            return image

    def plot_mge(self):

        image, x, y = self._get_mge()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("RA [arcsec]")
        ax.set_ylabel("DEC [arcsec]")
        im = plt.imshow(log10(image.T), extent=(npmin(x), npmax(x), npmin(y), npmax(y)))
        colorbar = fig.colorbar(im)
        colorbar.set_label(r'$\log\Sigma/L_\odot * \,{\rm pc}^2$')
        plt.show()

    def plot_data_kinematics(self, **kwargs):

        # get data
        vel, sig, X, Y, bins, s, dx = self._get_kinematic_data()

        vel_image = self.display_pixels(X[s], Y[s], vel[bins[s]], pixelsize=dx)
        sig_image = self.display_pixels(X[s], Y[s], sig[bins[s]], pixelsize=dx)

        # plotting
        fig = plt.figure(figsize=(14, 7.5))
        ax = fig.add_subplot(121)
        ax.set_xlabel("RA [arcsec]")
        ax.set_ylabel("DEC [arcsec]")
        image = plt.imshow(vel_image, cmap=sauron, interpolation='nearest',
                           extent=[X[s].min() - dx, X[s].max() + dx,
                                   Y[s].min() - dx, Y[s].max() + dx], **kwargs)

        colorbar = fig.colorbar(image)
        colorbar.set_label(r'$v$ [km/s]')

        ax2 = fig.add_subplot(122)
        ax2.set_xlabel("RA [arcsec]")
        ax2.set_ylabel("DEC [arcsec]")
        image2 = plt.imshow(sig_image, cmap=sauron, interpolation='nearest',
                            extent=[X[s].min() - dx, X[s].max() + dx,
                                    Y[s].min() - dx, Y[s].max() + dx], **kwargs)
        colorbar = fig.colorbar(image2)
        colorbar.set_label(r'$\sigma$ [km/s]')
        plt.show()

    def plot_comparison_model(self, model, inclination=90, one_figure=True, save_fig=False,
                              reverse_v_field=False, PSF_correction=False, **kwargs):

        if isinstance(model, FJmodel):
            f = model
        elif isinstance(model, basestring):
            f = FJmodel(model)
        else:
            raise ValueError(" -- ERROR: either pass an FJmodel instance or full qualified path")

        # get data
        vel, sig, vel_err, sig_err, X, Y, bins, s, dx, minx, miny, nx, ny, xt, yt =\
            self._get_kinematic_data(full_output=True)

        vel_image = self.display_pixels(X[s], Y[s], vel[bins[s]], pixelsize=dx)
        sig_image = self.display_pixels(X[s], Y[s], sig[bins[s]], pixelsize=dx)

        # get MGE data
        mge = self._get_mge(xt=xt, yt=yt, angle=self.angle)

        # get model data
        vel_model, sig_model, density_model = self._get_model_kinematics(f, inclination,
                                                                         s, bins, xt, yt, nx, ny,
                                                                         PSF_correction=PSF_correction)

        vel_image_mod = self.display_pixels(X[s], Y[s], vel_model[bins[s]], pixelsize=dx)
        sig_image_mod = self.display_pixels(X[s], Y[s], sig_model[bins[s]], pixelsize=dx)

        # peaks of velocity moments, used for re-scaling the model
        '''
        data_scale = npmax(vel[bins[s]]), npmax(sig[bins[s]])
        model_scale = npmax(vel_model[bins[s]]), npmax(sig_model[bins[s]])
        '''

        '''
        data_scale = npmax(sqrt(vel[bins[s]] ** 2 + sig[bins[s]] ** 2)),\
            npmax(sqrt(vel[bins[s]] ** 2 + sig[bins[s]] ** 2))
        model_scale = npmax(sqrt(vel_model[bins[s]] ** 2 + sig_model[bins[s]] ** 2)),\
            npmax(sqrt(vel_model[bins[s]] ** 2 + sig_model[bins[s]] ** 2))
        '''
        data_scale = max(npmax(vel[bins[s]]), npmax(sig[bins[s]])),\
            max(npmax(vel[bins[s]]), npmax(sig[bins[s]]))
        model_scale = max(npmax(vel_model[bins[s]]), npmax(sig_model[bins[s]])),\
            max(npmax(vel_model[bins[s]]), npmax(sig_model[bins[s]]))

        # colour scales of the velocity and velocity dispersion plots
        vmin, vmax = npmin(vel[bins[s]]), npmax(vel[bins[s]])
        smin, smax = npmin(sig[bins[s]]), npmax(sig[bins[s]])

        data_contour_levels = linspace(float((log10(mge)).min()) * .6, 0, num=6)
        model_contour_levels = linspace(float(density_model.min()) * .6, 0, num=6)

        # do I have to reverse the Velocity field?
        if reverse_v_field:
            vel_image_mod = -vel_image_mod

        # plotting
        if one_figure:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(221)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.set_xlabel("RA [arcsec]")
        ax.set_ylabel("DEC [arcsec]")
        image = plt.imshow(vel_image, cmap=sauron, interpolation='nearest',
                           extent=[(X[s].min() - dx) * CALIFA_arcsec_spaxel, (X[s].max() + dx) * CALIFA_arcsec_spaxel,
                                   (Y[s].min() - dx) * CALIFA_arcsec_spaxel, (Y[s].max() + dx) * CALIFA_arcsec_spaxel],
                           **kwargs)

        image.set_clim(vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(image)
        colorbar.set_label(r'$v$ [km/s]')
        # add density contours
        ax.contour(xt * CALIFA_arcsec_spaxel, yt * CALIFA_arcsec_spaxel, log10(mge).T,
                   colors='k', levels=data_contour_levels)

        if one_figure:
            ax2 = fig.add_subplot(222)
        else:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
        ax2.set_xlabel("RA [arcsec]")
        ax2.set_ylabel("DEC [arcsec]")
        image2 = plt.imshow(sig_image, cmap=sauron, interpolation='nearest',
                            extent=[(X[s].min() - dx) * CALIFA_arcsec_spaxel, (X[s].max() + dx) * CALIFA_arcsec_spaxel,
                                    (Y[s].min() - dx) * CALIFA_arcsec_spaxel, (Y[s].max() + dx) * CALIFA_arcsec_spaxel],
                            **kwargs)

        image2.set_clim(vmin=smin, vmax=smax)
        if one_figure:
            colorbar = fig.colorbar(image2)
        else:
            colorbar = fig2.colorbar(image2)
        colorbar.set_label(r'$\sigma$ [km/s]')
        # add density contours
        ax2.contour(xt * CALIFA_arcsec_spaxel, yt * CALIFA_arcsec_spaxel, log10(mge).T,
                    log10(mge).T, colors='k', levels=data_contour_levels)

        if one_figure:
            ax3 = fig.add_subplot(223)
        else:
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
        ax3.set_xlabel("RA [arcsec]")
        ax3.set_ylabel("DEC [arcsec]")
        image3 = plt.imshow(vel_image_mod / model_scale[0] * data_scale[0], cmap=sauron, interpolation='nearest',
                            extent=[(X[s].min() - dx) * CALIFA_arcsec_spaxel, (X[s].max() + dx) * CALIFA_arcsec_spaxel,
                                    (Y[s].min() - dx) * CALIFA_arcsec_spaxel, (Y[s].max() + dx) * CALIFA_arcsec_spaxel],
                            **kwargs)

        image3.set_clim(vmin=vmin, vmax=vmax)
        if one_figure:
            colorbar = fig.colorbar(image3)
        else:
            colorbar = fig3.colorbar(image3)
        colorbar.set_label(r'$v$ [km/s]')
        # add density contours
        ax3.contour(xt * CALIFA_arcsec_spaxel, yt * CALIFA_arcsec_spaxel, density_model.T,
                    colors='k', levels=model_contour_levels)

        if one_figure:
            ax4 = fig.add_subplot(224)
        else:
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(111)
        ax4.set_xlabel("RA [arcsec]")
        ax4.set_ylabel("DEC [arcsec]")
        image4 = plt.imshow(sig_image_mod / model_scale[1] * data_scale[1], cmap=sauron, interpolation='nearest',
                            extent=[(X[s].min() - dx) * CALIFA_arcsec_spaxel, (X[s].max() + dx) * CALIFA_arcsec_spaxel,
                                    (Y[s].min() - dx) * CALIFA_arcsec_spaxel, (Y[s].max() + dx) * CALIFA_arcsec_spaxel],
                            **kwargs)
        image4.set_clim(vmin=smin, vmax=smax)
        if one_figure:
            colorbar = fig.colorbar(image4)
        else:
            colorbar = fig4.colorbar(image4)
        colorbar.set_label(r'$\sigma$ [km/s]')
        # add density contours
        ax4.contour(xt * CALIFA_arcsec_spaxel, yt * CALIFA_arcsec_spaxel, density_model.T,
                    colors='k', levels=model_contour_levels)

        # V_RMS Figure
        if one_figure:
            vrms_image, vrms_image_mod = sqrt(vel_image ** 2 + sig_image ** 2),\
                sqrt((vel_image_mod / model_scale[0] * data_scale[0]) ** 2 +
                     (sig_image_mod / model_scale[1] * data_scale[1]) ** 2)

            fig2 = plt.figure()
            ax = fig2.add_subplot(121)

            ax.set_xlabel("RA [arcsec]")
            ax.set_ylabel("DEC [arcsec]")
            image = plt.imshow(vrms_image, cmap=sauron, interpolation='nearest',
                               extent=[X[s].min() - dx, X[s].max() + dx,
                                       Y[s].min() - dx, Y[s].max() + dx], **kwargs)

            colorbar = fig2.colorbar(image)
            colorbar.set_label(r'$V_{\rm RMS}$ [km/s]')
            # add density contours
            ax.contour(xt, yt, log10(mge).T, colors='k', levels=data_contour_levels)

            ax2 = fig2.add_subplot(122)

            ax2.set_xlabel("RA [arcsec]")
            ax2.set_ylabel("DEC [arcsec]")
            image2 = plt.imshow(vrms_image_mod, cmap=sauron, interpolation='nearest',
                                extent=[X[s].min() - dx, X[s].max() + dx,
                                        Y[s].min() - dx, Y[s].max() + dx], **kwargs)

            colorbar = fig2.colorbar(image2)
            colorbar.set_label(r'$V_{\rm RMS}$ [km/s]')
            # add density contours
            ax2.contour(xt, yt, density_model.T, colors='k', levels=model_contour_levels)

        # Save figures

        if save_fig:
            if one_figure:
                fig.savefig("figure1.eps", bbox_inches='tight')
            else:
                fig.savefig("figure1.eps", bbox_inches='tight')
                fig2.savefig("figure2.eps", bbox_inches='tight')
                fig3.savefig("figure3.eps", bbox_inches='tight')
                fig4.savefig("figure4.eps", bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    def _get_model_kinematics(self, f, inclination, s, bins, xt, yt, nx, ny, PSF_correction=False):

        # get model
        x, y = f.project(inclination=inclination, nx=60, npsi=31, scale='log')

        # maxgrid: max value of the observed grid. Used to rescale the model image
        # maxgrid = max(max(npmax(xt), abs(npmin(xt))), max(npmax(yt), abs(npmin(yt))))

        '''
        sigma, velocity, density = SmoothBivariateSpline(X.flatten(), Y.flatten(), f.slos.flatten()),\
            SmoothBivariateSpline(X.flatten(), Y.flatten(), f.vlos.flatten()),\
            SmoothBivariateSpline(X.flatten(), Y.flatten(), f.dlos.flatten())
        '''
        sigma, velocity, density = interp2d(x, y, f.slos.T, kind='cubic'), interp2d(x, y, f.vlos.T, kind='cubic'), \
            interp2d(x, y, f.dlos.T, kind='cubic')
        sigma_model, velocity_model = zeros(nx * ny), zeros(nx * ny)
        sig_model, vel_model = zeros(npmax(bins[s]) + 1), zeros(npmax(bins[s]) + 1)
        density_model = zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                x_rotated = (cos(radians(self.angle)) * xt[i] - sin(radians(self.angle)) * yt[ny - 1 - j]) \
                    * float(f.r_eff / self.Re)  # Rmax / maxgrid
                y_rotated = (sin(radians(self.angle)) * xt[i] + cos(radians(self.angle)) * yt[ny - 1 - j]) \
                    * float(f.r_eff / self.Re)  # Rmax / maxgrid
                sigma_model[i * ny + j] = sigma(x_rotated, y_rotated)
                velocity_model[i * ny + j] = velocity(x_rotated, y_rotated)

                x_rotated = (cos(radians(self.angle)) * xt[i] - sin(radians(self.angle)) * yt[j]) \
                    * float(f.r_eff / self.Re)  # Rmax / maxgrid
                y_rotated = (sin(radians(self.angle)) * xt[i] + cos(radians(self.angle)) * yt[j]) \
                    * float(f.r_eff / self.Re)  # Rmax / maxgrid
                density_model[i, j] = density(x_rotated, y_rotated)

        # Compute PSF correction, if needed
        sigma_model_psf, velocity_model_psf = None, None
        if PSF_correction:
            sigma_model_psf = gaussian_filter(sigma_model.reshape((nx, ny)), 3.)  # 3" PSF (CALIFA~1"/pix)
            velocity_model_psf = gaussian_filter(velocity_model.reshape((nx, ny)), 3.)  # 3" PSF (CALIFA~1"/pix)
            sigma_model_psf, velocity_model_psf = sigma_model_psf.reshape(nx * ny), velocity_model_psf.reshape(nx * ny)

        # normalize density to its maximum
        density_model -= npmax(density_model)

        bins2 = numpy_reshape(bins, (ny, nx))
        bins_img = rot90(rot90(rot90(bins2)))
        bins2 = bins_img.reshape(bins_img.shape[0] * bins_img.shape[1])

        for i in range(max(bins[s]) + 1):
            w = where(bins2 == i)
            if PSF_correction:
                sig_model[i] = average(sigma_model_psf[w])  # / npmax(sigma_model))
                vel_model[i] = average(velocity_model_psf[w])  # / npmax(velocity_model))
            else:
                sig_model[i] = average(sigma_model[w])  # / npmax(sigma_model))
                vel_model[i] = average(velocity_model[w])  # / npmax(velocity_model))

        return vel_model, sig_model, density_model

    def plot_vel_profiles(self, **kwargs):

        # get data
        vel, sig, vel_err, sig_err, X, Y, bins, s, dx, minx, miny, nx, ny, xt, yt =\
            self._get_kinematic_data(full_output=True)

        vel_image = self.display_pixels(X[s], Y[s], vel[bins[s]], pixelsize=dx)

        plot = 1
        if plot:
            # plotting
            fig = plt.figure(figsize=(14, 7.5))
            ax = fig.add_subplot(111)
            ax.set_xlabel("RA [arcsec]")
            ax.set_ylabel("DEC [arcsec]")
            image = plt.imshow(vel_image, cmap=sauron, interpolation='nearest',
                               extent=[X[s].min() - dx, X[s].max() + dx,
                                       Y[s].min() - dx, Y[s].max() + dx],
                               **kwargs)

            colorbar = fig.colorbar(image)
            colorbar.set_label(r'$v$ [km/s]')

        xd, X_pv, X_xd_pv, x, y = self._get_vel_curve_idx(X, Y, s, dx, bins, vel, full_output=True)

        plt.plot(x, y, 'ko')
        plt.show()

        # plot position-velocity diagrams
        fig = plt.figure(figsize=(14, 7.5))
        ax = fig.add_subplot(121)
        ax.set_xlabel("semi-major axis [arcsec]")
        ax.set_ylabel("velocity [km/s]")
        ax.plot(X_pv, vel[bins[s]], 'b.')
        ax.errorbar(X_xd_pv, (vel[bins[s]])[xd], yerr=(vel_err[bins[s]])[xd], fmt='o', color='r')

        ax2 = fig.add_subplot(122)
        ax2.set_xlabel("semi-major axis [arcsec]")
        ax2.set_ylabel("velocity dispersion [km/s]")
        ax2.plot(X_pv, sig[bins[s]], 'b.')
        ax2.errorbar(X_xd_pv, (sig[bins[s]])[xd], yerr=(sig_err[bins[s]])[xd], fmt='o', color='r')
        plt.show()

    def plot_comparison_model_vel_profiles(self, model, inclination=90, save_fig=False):

        if isinstance(model, FJmodel):
            f = model
        elif isinstance(model, basestring):
            f = FJmodel(model)
        else:
            raise NotImplementedError(" -- ERROR: either pass an FJmodel instance or full qualified path")

        # get data
        vel, sig, vel_err, sig_err, X, Y, bins, s, dx, minx, miny, nx, ny, xt, yt =\
            self._get_kinematic_data(full_output=True)

        xd, X_pv, X_xd_pv = self._get_vel_curve_idx(X, Y, s, dx, bins, vel)

        x, y = f.project(inclination=inclination, nx=60, npsi=31, scale='log')

        # peaks of velocity moments, used for re-scaling the model
        data_scale = npmax(vel[bins[s]]), npmax(sig[bins[s]])
        model_scale = npmax(f.vlos), npmax(f.slos)

        id_min, id_max = npabs(x * float(self.Re / f.r_eff) - array(X_xd_pv).min()).argmin(),\
            npabs(x * float(self.Re / f.r_eff) - array(X_xd_pv).max()).argmin()

        # plot position-velocity diagrams
        fig = plt.figure(figsize=(14, 7.5))
        ax = fig.add_subplot(121)
        ax.set_xlabel("semi-major axis [arcsec]", fontsize=18)
        ax.set_ylabel("velocity [km/s]", fontsize=18)
        ax.plot(x[id_min:id_max] * float(self.Re / f.r_eff),
                f.vlos[id_min:id_max, len(f.vlos) / 2] / model_scale[0] * data_scale[0], 'b-')
        ax.errorbar(X_xd_pv, (vel[bins[s]])[xd], yerr=(vel_err[bins[s]])[xd], fmt='o', color='r', label=self.gal_name)

        ax2 = fig.add_subplot(122)
        ax2.set_xlabel("semi-major axis [arcsec]", fontsize=18)
        ax2.set_ylabel("velocity dispersion [km/s]", fontsize=18)
        ax2.plot(x[id_min:id_max] * float(self.Re / f.r_eff),
                 f.slos[id_min:id_max, len(f.slos) / 2] / model_scale[1] * data_scale[1], 'b-')
        ax2.errorbar(X_xd_pv, (sig[bins[s]])[xd], yerr=(sig_err[bins[s]])[xd], fmt='o', color='r', label=self.gal_name)

        if save_fig:
            ax.legend(loc='best')
            ax2.legend(loc='best')
            plt.savefig('vprof_mod_comp' + self.gal_name + '.eps', bbox_inches='tight')
        plt.show()

    def _get_flux_bin(self):

        # read flux data from fits file
        hdu = pyfits.open(self.fits_file)
        d = hdu[1].data

        flux = []

        for i in range(len(d)):
            flux.append(d[i][7])

        # flux = -1. / 2.5 * array(flux)
        flux = array(flux)

        return flux

    def _get_flux_contour(self):

        flux = self._get_flux_bin()

        # get kinematic data
        vel, sig, X, Y, bins, s, dx = self._get_kinematic_data()

        minx, miny, step_x, step_y, nx, ny, dx = self._read_aperture_file()
        xx = linspace(X[s].min(), X[s].max(), num=int(X[s].max() - X[s].min()) + 1)
        yy = linspace(Y[s].min(), Y[s].max(), num=int(Y[s].max() - Y[s].min()) + 1)

        flux_contour = full((nx, ny), nan)
        for i in range(len(xx)):
            for j in range(len(yy)):

                ww = ((npabs(around(array(X[s]) - xx[i], decimals=3)) < 0.01) &
                      (npabs(around(array(Y[s]) - yy[j], decimals=3)) < 0.01))

                # if list is empty
                if not (flux[s])[ww]:
                    pass
                else:
                    flux_contour[i, j] = (flux[s])[ww]

        return flux_contour

    def plot_light_profile(self, Re_fix=None, model=None, inclination=90, Re_model=None, nx=100,
                           save_fig=False, **kwargs):

        # plotting surface brightness profile with Sersic fits
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.get_sb_profile(Re_fix=Re_fix, show=False)

        if model is not None and isinstance(model, FJmodel):

            f = model

            if Re_model is None:
                print 'Projecting model to compute Re...'
                f.project(inclination=inclination, nx=60, scale='log', verbose=False)
                Re_model = f.r_eff

            r_mod, gc_mod = f.light_profile(inclination=inclination, nx=nx, npsi=31,
                                            Re_model=Re_model, Re_data=self.Re,
                                            xmin=self.R_arcsec[0], xmax=self.R_arcsec[-1], num=len(self.R_arcsec),
                                            **kwargs)

            gc_scale = self.gc[npabs(self.R_arcsec - self.Re).argmin()] - gc_mod[npabs(r_mod - self.Re).argmin()]
            R_mod, SB_mod = KinData.get_surface_brightness(r_mod, gc_mod + gc_scale)
            if Re_fix is not None:
                ax.plot(R_mod / Re_fix, SB_mod / SB_mod[npabs(R_mod - Re_fix).argmin()], 'bo',
                        label=r"$f(\bf J)$ model")
            else:
                ax.plot(R_mod / self.Re, SB_mod / SB_mod[npabs(R_mod - self.Re).argmin()], 'bo',
                        label=r"$f(\bf J)$ model")
            ax.set_xlabel(r"$R/R_{\rm e}$", fontsize=18)
            ax.set_ylabel(r"$I(R)/I(R_{\rm e})$", fontsize=18)
            ax.set_xscale('log')

            # Plot of the growth curves
            # plt.figure()
            # plt.plot(r_mod, gc_mod + gc_scale, 'bo', self.R_arcsec, self.gc, 'ro')
            plt.legend(loc='best', fontsize=16)
            if save_fig:
                plt.savefig(self.gal_name + "_SB_wmodel.pdf", bbox_inches='tight')
            plt.show()

    def get_sb_profile(self, Re_fix=None, show=True, **kwargs):

        R, sb = self.get_surface_brightness(self.R_arcsec, self.gc)
        Re, n, I_0, Re_fix, n_fix, I_0_fix = self.fit_sersic_profile(R, -sb, Re_fix=Re_fix, **kwargs)

        plt.gca().invert_yaxis()

        if Re_fix is not None:
            plt.plot(R / Re_fix, sb / sb[npabs(R - Re_fix).argmin()], 'ro-', label=self.gal_name)
            plt.plot(R / Re_fix, KinData.sersic(R, Re_fix, n_fix, I_0_fix) /
                     KinData.sersic(Re_fix, Re_fix, n_fix, I_0_fix), 'k-', lw=2, label=u'Sérsic, n=%2.1f' % n_fix)
        else:
            plt.plot(R / Re, sb / sb[npabs(R - Re).argmin()], 'ro-', label=self.gal_name)
            plt.plot(R / Re, KinData.sersic(R, Re, n, I_0) /
                     KinData.sersic(Re, Re, n, I_0), 'k-', lw=2, label=u'Sérsic, n=%2.1f' % n)

        if show:
            plt.show()

    @staticmethod
    def display_pixels(x, y, val, pixelsize=None, angle=None):
        """
        From M. Cappellari's analogous routine.
        Display vectors of square pixels at coordinates (x,y) coloured with "val".
        An optional rotation angle can be applied to the pixels.

        """
        if pixelsize is None:
            pixelsize = npmin(distance.pdist(column_stack([x, y])))

        if hasattr(x, 'min') and hasattr(x, 'max') and hasattr(y, 'min') and hasattr(y, 'max'):
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
        else:
            raise TypeError("x or y are not arrays")

        nx = round((xmax - xmin) / pixelsize) + 1
        ny = round((ymax - ymin) / pixelsize) + 1
        img = empty((nx, ny)) + nan  # 1e-10
        j = round((x - xmin) / pixelsize).astype(int)
        k = round((y - ymin) / pixelsize).astype(int)
        img[j, k] = val

        if (angle is None) or (angle == 0):

            return rot90(img)

        else:

            xx = linspace(xmin - pixelsize / 2, xmax + pixelsize / 2, nx + 1)
            yy = linspace(ymin - pixelsize / 2, ymax + pixelsize / 2, ny + 1)
            x, y = meshgrid(xx, yy)
            x, y = x.T, y.T

            cpa = cos(radians(angle))
            spa = sin(radians(angle))
            x, y = x * cpa - y * spa, x * spa + y * cpa

            mask = ones_like(img)
            mask[j, k] = 0
            img = masked_array(img, mask=mask)

            mask1 = ones_like(x)
            mask1[:-1, :-1] *= mask  # flag the four corners of the mesh
            mask1[:-1, 1:] *= mask
            mask1[1:, :-1] *= mask
            mask1[1:, 1:] *= mask
            x = masked_array(x, mask=mask1)  # Mask is used for proper plot range
            y = masked_array(y, mask=mask1)

            return img, x, y

            # f = ax.pcolormesh(x, y, img, cmap=sauron, **kwargs)
            # ax.axis('image')

    def _get_vel_curve_idx(self, X, Y, s, dx, bins, vel, full_output=False):

        theta, x, y = self.get_major_axis(X[s], Y[s], vel[bins[s]])

        # plt.plot([(X[s])[id_max], (X[s])[id_min]], [(Y[s])[id_max], (Y[s])[id_min]], 'ms')

        # compute distance of each spaxel from the major axis line
        dist = (distance.cdist(dstack([y, x])[0], dstack([Y[s], X[s]])[0]))
        xd = []
        for i in range(len(dist)):
            xd.append(dist[i].argmin())

        # compute distance (with sign) from centre of map
        X_pv, X_xd_pv = [], []
        d_pv, d_xd_pv = distance.cdist([(0., 0.)], dstack([Y[s], X[s]])[0])[0],\
            distance.cdist([(0., 0.)], dstack([(Y[s])[xd], (X[s])[xd]])[0])[0]

        for i in range(len(d_pv)):
            if (vel[bins[s]])[i] > 0.:
                X_pv.append(d_pv[i])
            else:
                X_pv.append(-d_pv[i])

        for i in range(len(d_xd_pv)):
            if ((vel[bins[s]])[xd])[i] > 0.:
                X_xd_pv.append(d_xd_pv[i])
            else:
                X_xd_pv.append(-d_xd_pv[i])

        if full_output:
            return xd, X_pv, X_xd_pv, x, y
        else:
            return xd, X_pv, X_xd_pv

    @staticmethod
    def get_major_axis(X, Y, v):

        xmin, xmax = X.min(), X.max()
        thetas, x = linspace(-pi / 3., pi / 3., num=100), linspace(xmin, xmax, num=100)

        dv, theta_out = 0., 0.
        for theta in thetas:
            m = tan(theta)
            y = m * x

            # compute distance of each spaxel from the major axis line
            dist = (distance.cdist(dstack([y, x])[0], dstack([Y, X])[0]))
            xd = []
            for i in range(len(dist)):
                xd.append(dist[i].argmin())

            delta = npabs((v[xd])[0:len(xd) / 2 - 1] - (v[xd])[len(xd) / 2:-1])
            if delta.mean() > dv:
                dv = delta.mean()
                theta_out = theta

        return theta_out, x, tan(theta_out) * x

    @staticmethod
    def get_surface_brightness(R, gc):

        gc = 10. ** (-0.4 * gc)  # convert into fluxes
        sb, Rout = [-2.5 * log10(gc[0] / (4. * pi * R[0] ** 2))], [R[0]]

        for i in range(1, len(R)):
            area = 4. * pi * (R[i] ** 2 - R[i - 1] ** 2)
            sb.append(-2.5 * log10(gc[i] - gc[i - 1]) + 2.5 * log10(area))
            Rout.append(R[i])

        return array(Rout), array(sb)

    @staticmethod
    def sersic(x, Re, n, I_0):
        x = asarray(x)
        b = 2. * n - 1. / 3. + 4. / (405. * n)
        return 2.5 * log10(I_0 * exp(- b * (x / abs(Re)) ** (1. / n)))

    @staticmethod
    def fit_sersic_profile(R, sb, I_0_in=None, Re_fix=None):

        Re_min, n_min, I_0_min = None, None, None
        n_fix, I_0_fix = None, None
        if I_0_in is None:
            I_0_in = 10. ** (0.4 * sb[0])

        if Re_fix is None:
            '''
                Likelihood with fixed effective radius
            '''
            def likelihood(t, x, y):
                # chi squared likelihood
                n, Re, I_0 = t
                chi_square = (y - KinData.sersic(x, Re, n, I_0)) ** 2
                return chi_square[chi_square < 999.].sum()
        else:
            '''
                Likelihood with variable effective radius
            '''
            def likelihood(t, x, y, Re):
                # chi squared likelihood
                n, I_0 = t
                chi_square = (y - KinData.sersic(x, Re, n, I_0)) ** 2
                return chi_square[chi_square < 999.].sum()

        f = lambda *args: likelihood(*args)
        if Re_fix is None:
            # in method L-BFGS-B bounds for the parameters can be specified
            res = minimize(f, [4., 10., I_0_in], args=(R, sb),
                           method='L-BFGS-B', bounds=[(0.5, 20.), (.1, 100.), (I_0_in * 1e-1, I_0_in * 1e3)])
            n_min, Re_min, I_0_min = res["x"]
            chi_sq_min = (sb - KinData.sersic(R, Re_min, n_min, I_0_min)) ** 2
            print "\nLikelihood minimization: \t (%f, %f, %e) \t CHIsq=%f" % \
                (Re_min, n_min, I_0_min, chi_sq_min[chi_sq_min < 999.].sum())
        else:
            # in method L-BFGS-B bounds for the parameters can be specified
            res = minimize(f, [4., I_0_in], args=(R, sb, Re_fix),
                           method='L-BFGS-B', bounds=[(0.5, 20.), (I_0_in * 1e-1, I_0_in * 1e3)])
            n_fix, I_0_fix = res["x"]
            chi_sq_min = (sb - KinData.sersic(R, Re_fix, n_fix, I_0_fix)) ** 2
            print "\nLikelihood minimization (w. fixed Re): \t (%f, %f, %e) \t CHIsq=%f" % \
                (Re_fix, n_fix, I_0_fix, chi_sq_min[chi_sq_min < 999.].sum())

        try:
            '''
                Try with the curve_fit method.
                Usually gives better results in terms of ChiSQ, so its output is preferred
            '''
            w = (sb < 999.) & (sb > -999.)
            p_opt, p_cov = curve_fit(KinData.sersic, R[w], sb[w], p0=[10., 4., I_0_in])
            Re_c, n_c, I_0_c = p_opt[0], p_opt[1], p_opt[2]
            chi_sq_c = (sb - KinData.sersic(R, Re_c, n_c, I_0_c)) ** 2
            print "\nCurve fitting: \t (%f, %f, %e) \t CHIsq=%f" % \
                  (Re_c, n_c, I_0_c, chi_sq_c[chi_sq_c < 999.].sum())

            '''
                Return Values
            '''
            if Re_fix is None:
                ret = Re_c, n_c, I_0_c, Re_min, n_min, I_0_min
            else:
                ret = Re_c, n_c, I_0_c, Re_fix, n_fix, I_0_fix

        except RuntimeError:
            if Re_fix is None:
                ret = Re_min, n_min, I_0_min, None, None, None
            else:
                ret = Re_fix, n_fix, I_0_fix, None, None, None

        return ret

    def _get_vu(self, X, Y, vel, sig):

        # get MGE data
        dat = genfromtxt(self.mge_file)
        sb = dat[:, 0]
        sigma = dat[:, 1]
        q = dat[:, 2]

        R = sqrt(power(X, 2) + power(Y, 2))
        dR = gradient(R)
        vu = 0.
        for i in range(len(vel)):
            """
            do a proper Sigma(R) Vrms(R) RdR integral...
            """

            v_rms = sqrt(vel[i] ** 2 + sig[2] ** 2)
            vu += R[i] * v_rms * dR

    def _get_kinematic_data(self, full_output=False):

        # get bins
        bins, s = self._get_bins()

        # read aperture file
        minx, miny, step_x, step_y, nx, ny, dx = self._read_aperture_file()

        # get X, Y
        xt, yt = None, None
        if full_output:
            X, Y, xt, yt = self._get_voronoi_coordinates(minx, miny, dx, nx, ny, angle=0., full_output=full_output)
        else:
            X, Y = self._get_voronoi_coordinates(minx, miny, dx, nx, ny, angle=0.)

        # get kinematic data
        kin_data = genfromtxt(self.kin_data_file, skip_header=1)

        vel, sig = kin_data[:, 1], kin_data[:, 3]
        vel_err, sig_err = kin_data[:, 2], kin_data[:, 4]

        if full_output:
            return vel, sig, vel_err, sig_err, X, Y, bins, s, dx, minx, miny, nx, ny, xt, yt
        else:
            return vel, sig, X, Y, bins, s, dx

    def _get_bins(self):

        # read voronoi bins information
        bins = genfromtxt(self.bins_file, dtype=int)

        bins -= 1  # ATTENTION: here the bin number is shifted by -1 because of Fortran/C different index conventions
        s = where(bins > -0.5)[0]

        return bins, s

    def _get_effective_radius(self):

        # read CALIFA_ETGs.conf file
        for n_line in range(29, 39):
            line = getline(self.conf_file, n_line)

            if line.split()[1] == self.gal_name:
                Re = float(line.split()[16])

        return Re

    def _read_growth_curve_file(self):

        # read growth curve file
        gc_data = genfromtxt(self.gc_file)

        return gc_data[:, 0], gc_data[:, 2]

    def _read_aperture_file(self):

        # read aperture file
        line = getline(self.aperture_file, 2)
        minx, miny = float(line.split()[0]), float(line.split()[1])

        line = getline(self.aperture_file, 3)
        step_x, step_y = float(line.split()[0]), float(line.split()[1])

        line = getline(self.aperture_file, 5)
        nx, ny = int(line.split()[0]), int(line.split()[1])

        dx = float(step_x / nx)

        return minx, miny, step_x, step_y, nx, ny, dx

    def _get_voronoi_coordinates(self, minx, miny, dx, nx, ny, angle=None, full_output=False):

        if angle is None:
            angle = self.angle

        # generate x, y
        xt = linspace(minx + dx / 2., dx * (nx - 1) + minx + dx / 2., num=nx)
        yt = linspace(miny + dx / 2., dx * (ny - 1) + miny + dx / 2., num=ny)
        Xt, Yt = meshgrid(xt, yt)

        # xtx = Xt.reshape(Xt.shape[0]*Xt.shape[1])
        # yty = Yt.reshape(Yt.shape[0]*Yt.shape[1])

        Xi = numpy_cos(radians(angle)) * Xt - numpy_sin(radians(angle)) * Yt
        Yi = numpy_sin(radians(angle)) * Xt + numpy_cos(radians(angle)) * Yt
        xi = Xi.reshape(Xi.shape[0] * Xi.shape[1])
        yi = Yi.reshape(Yi.shape[0] * Yi.shape[1])

        if full_output:
            return xi, yi, xt, yt
        else:
            return xi, yi
