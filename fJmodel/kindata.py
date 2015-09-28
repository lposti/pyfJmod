# -*- coding: utf-8 -*-
__author__ = 'lposti'

from fJmodel import FJmodel
from warnings import warn
from sauron import sauron
from numpy import genfromtxt, zeros, linspace, exp, log10, column_stack, round, empty, nan, rot90, \
    meshgrid, radians, ones_like, where, average, dstack, array, full, around, \
    asarray, isinf, isnan, sqrt
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
from scipy.interpolate import interp2d, interp1d
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
                self.conf_file = directory[:-8] + 'conf/CALIFA_ETGs.conf'
                self.conf_mass_file = directory[:-8] + 'conf/califa4dfmodels.dat'
                self.conf_kpcarcsec_file = directory[:-8] + 'conf/CALIFA_Lorenzo.conf'
            else:
                self.gal_name = directory[-7:]
                self.conf_file = directory[:-7] + 'conf/CALIFA_ETGs.conf'
                self.conf_mass_file = directory[:-7] + 'conf/califa4dfmodels.dat'
                self.conf_kpcarcsec_file = directory[:-7] + 'conf/CALIFA_Lorenzo.conf'
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

        self.pa, self.pa_err = 0., 0.  # get through aperture file
        self.Re = self._get_effective_radius()
        try:
            self.m_star_tot, self.s_eff, self.m_dyn_Re, self.m_schw = self._get_masses()
            self.kpc_arcsec = self._get_kpc_over_arcsec()
        except UnboundLocalError:
            warn("\n--\n-- For this galaxy I still don't have masses and other global parameters\n--\n", Warning)
        self.angle = 90. - float(getline(self.aperture_file, 6).split()[0])
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
        vel_model, sig_model, dens_contour, dens_model = self._get_model_kinematics(f, inclination,
                                                                                    s, bins, xt, yt, nx, ny,
                                                                                    PSF_correction=PSF_correction,
                                                                                    density_bins=True)

        # shift the systemic velocity
        xd, X_pv, X_xd_pv = self._get_vel_curve_idx(X, Y, s, bins, vel)
        vel[bins[s]] = self._shift_systemic_velocity(vel[bins[s]], xd)

        vel_image_mod = self.display_pixels(X[s], Y[s], vel_model[bins[s]], pixelsize=dx)
        sig_image_mod = self.display_pixels(X[s], Y[s], sig_model[bins[s]], pixelsize=dx)

        '''
            Computing average sigma and lambda_R within effective radius
        '''
        s_re, s_mod_re, d_re = [], [], []
        lr_n, lr_d, v_sq, s_sq = [], [], [], []
        for t in s:
            if X[t] ** 2 + Y[t] ** 2 <= self.Re ** 2:
                d_re.append(10. ** dens_model[bins[t]])
                s_re.append((vel[bins[t]] ** 2 + sig[bins[t]] ** 2) ** 0.5)
                s_mod_re.append((vel_model[bins[t]] ** 2 + sig_model[bins[t]] ** 2) ** 0.5)

                v_sq.append(vel_model[bins[t]] ** 2)
                s_sq.append(sig_model[bins[t]] ** 2)
                lr_n.append(10. ** dens_model[bins[t]] * sqrt(X[t] ** 2 + Y[t] ** 2) * npabs(vel_model[bins[t]]))
                lr_d.append(10. ** dens_model[bins[t]] * sqrt(X[t] ** 2 + Y[t] ** 2) * sqrt(vel_model[bins[t]] ** 2 +
                                                                                            sig_model[bins[t]] ** 2))

        print
        print "lambda_R:", array(lr_n).sum() / array(lr_d).sum()
        print "V/Sigma: ", sqrt(average(v_sq, weights=d_re) / average(s_sq, weights=d_re))
        print
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''

        # peaks of velocity moments, used for re-scaling the model
        # data_scale = average(s_re), average(s_re)
        # model_scale = average(s_mod_re), average(s_mod_re)
        data_scale = average(s_re, weights=d_re), average(s_re, weights=d_re)
        model_scale = average(s_mod_re, weights=d_re), average(s_mod_re, weights=d_re)

        # colour scales of the velocity and velocity dispersion plots
        vmin, vmax = npmin(vel[bins[s]]), npmax(vel[bins[s]])
        smin, smax = npmin(sig[bins[s]]), npmax(sig[bins[s]])

        data_contour_levels = linspace(float((log10(mge)).min()) * .6, 0, num=6)
        model_contour_levels = linspace(float(dens_contour.min()) * .6, 0, num=6)

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
        ax3.contour(xt * CALIFA_arcsec_spaxel, yt * CALIFA_arcsec_spaxel, dens_contour.T,
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
        ax4.contour(xt * CALIFA_arcsec_spaxel, yt * CALIFA_arcsec_spaxel, dens_contour.T,
                    colors='k', levels=model_contour_levels)

        # V_RMS or V/sigma Figure
        if one_figure:
            # vrms_image, vrms_image_mod = sqrt(vel_image ** 2 + sig_image ** 2),\
            #     sqrt((vel_image_mod / model_scale[0] * data_scale[0]) ** 2 +
            #          (sig_image_mod / model_scale[1] * data_scale[1]) ** 2)
            vs_image, vs_image_mod = npabs(vel_image) / sig_image, npabs(vel_image_mod) / sig_image_mod

            vsmin, vsmax = npmin(npabs(vel[bins[s]]) / sig[bins[s]]), npmax(npabs(vel[bins[s]]) / sig[bins[s]])

            fig2 = plt.figure()
            ax = fig2.add_subplot(121)

            ax.set_xlabel("RA [arcsec]")
            ax.set_ylabel("DEC [arcsec]")
            image5 = plt.imshow(vs_image, cmap=sauron, interpolation='nearest',
                                extent=[X[s].min() - dx, X[s].max() + dx,
                                        Y[s].min() - dx, Y[s].max() + dx], **kwargs)

            image5.set_clim(vmin=vsmin, vmax=vsmax)
            colorbar = fig2.colorbar(image5)
            colorbar.set_label(r'$V_{\rm RMS}$ [km/s]')
            # add density contours
            ax.contour(xt, yt, log10(mge).T, colors='k', levels=data_contour_levels)

            ax2 = fig2.add_subplot(122)

            ax2.set_xlabel("RA [arcsec]")
            ax2.set_ylabel("DEC [arcsec]")
            image6 = plt.imshow(vs_image_mod, cmap=sauron, interpolation='nearest',
                                extent=[X[s].min() - dx, X[s].max() + dx,
                                        Y[s].min() - dx, Y[s].max() + dx], **kwargs)

            image6.set_clim(vmin=vsmin, vmax=vsmax)
            colorbar = fig2.colorbar(image6)
            colorbar.set_label(r'$V_{\rm RMS}$ [km/s]')
            # add density contours
            ax2.contour(xt, yt, dens_contour.T, colors='k', levels=model_contour_levels)

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

    def _get_model_kinematics(self, f, inclination, s, bins, xt, yt, nx, ny, PSF_correction=False, density_bins=False):

        # get model
        x, y = f.project(inclination=inclination, nx=60, scale='log')

        sigma, velocity, density = interp2d(x, y, f.slos.T, kind='cubic'), interp2d(x, y, f.vlos.T, kind='cubic'), \
            interp2d(x, y, f.dlos.T, kind='cubic')
        sigma_model, velocity_model, density_model = zeros(nx * ny), zeros(nx * ny), zeros(nx * ny)
        sig_model, vel_model, dens_model = zeros(npmax(bins[s]) + 1), zeros(npmax(bins[s]) + 1), \
            zeros(npmax(bins[s]) + 1)
        dens_contour = zeros((nx, ny))

        x_eps, y_eps = [], []
        for i in range(nx):
            for j in range(ny):
                x_rotated = (cos(radians(self.angle)) * xt[i] - sin(radians(self.angle)) * yt[ny - 1 - j]) \
                    * float(f.r_eff / self.Re)  # Rmax / maxgrid
                y_rotated = (sin(radians(self.angle)) * xt[i] + cos(radians(self.angle)) * yt[ny - 1 - j]) \
                    * float(f.r_eff / self.Re)  # Rmax / maxgrid
                sigma_model[i * ny + j] = sigma(x_rotated, y_rotated)
                velocity_model[i * ny + j] = velocity(x_rotated, y_rotated)
                density_model[i * ny + j] = density(x_rotated, y_rotated)
                x_eps.append(x_rotated)
                y_eps.append(y_rotated)

                x_rotated = (cos(radians(self.angle)) * xt[i] - sin(radians(self.angle)) * yt[j]) \
                    * float(f.r_eff / self.Re)  # Rmax / maxgrid
                y_rotated = (sin(radians(self.angle)) * xt[i] + cos(radians(self.angle)) * yt[j]) \
                    * float(f.r_eff / self.Re)  # Rmax / maxgrid
                dens_contour[i, j] = density(x_rotated, y_rotated)

        w = array(y_eps) ** 2 + array(x_eps) ** 2 <= self.Re ** 2
        print
        print "ellipticity: ", 1. - sqrt(average(array(y_eps)[w] ** 2, weights=10. ** density_model[w]) /
                                         average(array(x_eps)[w] ** 2, weights=10. ** density_model[w]))
        print
        # Compute PSF correction, if needed
        sigma_model_psf, velocity_model_psf = None, None
        if PSF_correction:
            # 2.7" FWHM-PSF (CALIFA~1"/pix)
            # for a gaussian: FWHM =~ 2.3548 sigma
            sigma_model_psf = gaussian_filter(sigma_model.reshape((nx, ny)), 2.7 / 2.3548)
            velocity_model_psf = gaussian_filter(velocity_model.reshape((nx, ny)), 2.7 / 2.3548)
            sigma_model_psf, velocity_model_psf = sigma_model_psf.reshape(nx * ny), velocity_model_psf.reshape(nx * ny)

        bins2 = numpy_reshape(bins, (ny, nx))
        bins_img = rot90(rot90(rot90(bins2)))
        bins2 = bins_img.reshape(bins_img.shape[0] * bins_img.shape[1])

        for i in range(max(bins[s]) + 1):
            w = where(bins2 == i)

            ''' Luminosity weighted mean '''
            dens_model[i] = average(density_model[w])
            if PSF_correction:
                sig_model[i] = average(sigma_model_psf[w], weights=10. ** density_model[w])
                vel_model[i] = average(velocity_model_psf[w], weights=10. ** density_model[w])
            else:
                sig_model[i] = average(sigma_model[w], weights=10. ** density_model[w])
                vel_model[i] = average(velocity_model[w], weights=10. ** density_model[w])

        if density_bins:
            return vel_model, sig_model, dens_contour, dens_model
        else:
            return vel_model, sig_model, dens_contour

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

        xd, X_pv, X_xd_pv, x, y = self._get_vel_curve_idx(X, Y, s, bins, vel, full_output=True)

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

    def plot_comparison_model_vel_profiles(self, model, inclination=90, save_fig=False, reverse_v_field=False,
                                           PSF_correction=False, order='row'):

        if isinstance(model, FJmodel):
            f = model
        elif isinstance(model, basestring):
            f = FJmodel(model)
        else:
            raise NotImplementedError(" -- ERROR: either pass an FJmodel instance or full qualified path")

        # get data
        vel, sig, vel_err, sig_err, X, Y, bins, s, dx, minx, miny, nx, ny, xt, yt =\
            self._get_kinematic_data(full_output=True)

        # vel[bins[s]] += 30.

        # get model data
        vel_model, sig_model, dens_contour, dens_model = self._get_model_kinematics(f, inclination,
                                                                                    s, bins, xt, yt, nx, ny,
                                                                                    PSF_correction=PSF_correction,
                                                                                    density_bins=True)

        '''
            Computing average sigma within effective radius
        '''
        s_re, s_mod_re, d_re = [], [], []
        for t in s:
            if X[t] ** 2 + Y[t] ** 2 <= (self.Re / 2) ** 2:
                d_re.append(10. ** dens_model[bins[t]])
                s_re.append((vel[bins[t]] ** 2 + sig[bins[t]] ** 2) ** 0.5)
                s_mod_re.append((vel_model[bins[t]] ** 2 + sig_model[bins[t]] ** 2) ** 0.5)
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''

        # do I have to reverse the Velocity field?
        if reverse_v_field:
            vel_model = -vel_model

        xd, X_pv, X_xd_pv = self._get_vel_curve_idx(X, Y, s, bins, vel)

        # rescale the models
        data_scale = average(s_re, weights=d_re), average(s_re, weights=d_re)
        model_scale = average(s_mod_re, weights=d_re), average(s_mod_re, weights=d_re)
        # data_scale = average(s_re), average(s_re)
        # model_scale = average(s_mod_re), average(s_mod_re)

        vel[bins[s]] = self._shift_systemic_velocity(vel[bins[s]], xd)

        '''
        plot position-velocity diagrams
        '''
        fig, ax, ax2 = None, None, None
        if order is 'row':
            fig = plt.figure(figsize=(14, 7.5))
            ax = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        elif order is 'column':
            fig = plt.figure(figsize=(9, 10))
            ax = fig.add_axes((.1, .5, .8, .4))
            ax2 = fig.add_axes((.1, .1, .8, .4))
        else:
            raise ValueError("order must be either 'row' or 'column'.")
        ax.set_xlabel(r"$\rm major \, axis \, [arcsec]$", fontsize=18)
        ax.set_ylabel(r"$\rm velocity \, [km/s]$", fontsize=18)

        x, y = f.project(inclination=inclination, nx=60, scale='log')

        slos_psf, vlos_psf = True, True
        X_mod = linspace(npmin(X_xd_pv), npmax(X_xd_pv), num=100)
        if PSF_correction:
            # 2.7" FWHM-PSF (CALIFA~1"/pix)
            # for a gaussian: FWHM =~ 2.3548 sigma
            slos_log, vlos_log = f.slos[:, len(x) / 2], f.vlos[:, len(x) / 2]
            f_interp = interp1d(x * float(self.Re / f.r_eff), slos_log)
            slos_lin = f_interp(X_mod)
            slos_psf = gaussian_filter(slos_lin, 2.7 * 1.6)
            f_interp = interp1d(x * float(self.Re / f.r_eff), vlos_log)
            vlos_lin = f_interp(X_mod)
            vlos_psf = gaussian_filter(vlos_lin, 2.7 * 1.6)

        # all bins
        # ax.errorbar(X_pv, vel[bins[s]], yerr=vel_err[bins[s]], fmt='.', color='#C3C3EC')

        ax.plot(X_xd_pv, (vel_model[bins[s]])[xd] / model_scale[0] * data_scale[0], 'sr',
                label=r"$f(\bf J) \, {\rm model}$")
        ax.plot(X_mod, vlos_psf / model_scale[0] * data_scale[0], 'r')
        ax.errorbar(X_xd_pv, (vel[bins[s]])[xd], yerr=(vel_err[bins[s]])[xd], fmt='o', color='b',
                    label=r"$\rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$")
        ax.set_xlim([npmin(X_xd_pv) * 1.1, npmax(X_xd_pv) * 1.1])

        ax2.set_xlabel(r"$\rm major \, axis \, [arcsec]$", fontsize=18)
        ax2.set_ylabel(r"$\rm velocity \, dispersion \, [km/s]$", fontsize=18)

        # all bins
        # ax2.errorbar(X_pv, sig[bins[s]], yerr=sig_err[bins[s]], fmt='.', color='#C3C3EC')

        ax2.plot(X_xd_pv, (sig_model[bins[s]])[xd] / model_scale[1] * data_scale[1],
                 'sr', label=r"$f(\bf J) \, {\rm model}$")

        ax2.plot(X_mod, slos_psf / model_scale[0] * data_scale[0], 'r')
        ax2.errorbar(X_xd_pv, (sig[bins[s]])[xd], yerr=(sig_err[bins[s]])[xd], fmt='o', color='b',
                     label=r"$\rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$")
        ax2.set_xlim([npmin(X_xd_pv) * 1.1, npmax(X_xd_pv) * 1.1])

        ax.legend(loc='best', fontsize=16)
        if order is 'column':
            ax.set_xticklabels([])
            ax2.set_ylim([(sig[bins[s]])[xd].min() * 0.25, (sig[bins[s]])[xd].max() * 1.2])
        else:
            ax2.legend(loc='best', fontsize=16)

        if save_fig:
            plt.savefig('vprof_mod_comp' + self.gal_name + '_PSF.pdf', bbox_inches='tight')
        plt.show()

    def plot_mass_profile(self, model, inclination=90., PSF_correction=True, save_fig=False):

        if isinstance(model, FJmodel):
            f = model
        elif isinstance(model, basestring):
            f = FJmodel(model)
        else:
            raise NotImplementedError(" -- ERROR: either pass an FJmodel instance or full qualified path")

        # get data
        vel, sig, vel_err, sig_err, X, Y, bins, s, dx, minx, miny, nx, ny, xt, yt =\
            self._get_kinematic_data(full_output=True)

        # get model data
        vel_model, sig_model, dens_contour = self._get_model_kinematics(f, inclination,
                                                                        s, bins, xt, yt, nx, ny,
                                                                        PSF_correction=PSF_correction)

        data_scale = max(npmax(vel[bins[s]]), npmax(sig[bins[s]]))
        model_scale = max(npmax(vel_model[bins[s]]), npmax(sig_model[bins[s]]))
        v_scale = data_scale / model_scale
        r_scale = self.Re * self.kpc_arcsec / f.r_eff

        G = 4.302e-3 * 1e-3  # 4.302e-3 pc Mo^-1 (km/s)^2 to kpc Mo^-1 (km/s)^2
        M_scale = v_scale ** 2 * r_scale / G

        print
        print "Estimated total mass:", f.m[-1] * M_scale, "5 R_e^2 sigma_e / G:", \
            5. * self.s_eff ** 2 * self.Re * self.kpc_arcsec / G

        w_max = npabs(f.ar - 20. * f.r_eff).argmin()
        plt.axvline(1., color='k', ls='--')
        plt.loglog(f.ar[:w_max] / f.r_eff, f.m[:w_max] * M_scale, 'r-', lw=2)
        plt.loglog(f.ar[w_max - 1] / f.r_eff, f.m[w_max - 1] * M_scale, 'rp',
                   label=r"$f(\bf J) \, \rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$", markersize=15)
        plt.loglog(1., 10. ** self.m_schw, 'kD', label=r"$M_{\rm Schw}(R_{\rm e})$", markersize=12)
        plt.loglog(f.ar[w_max - 1] / f.r_eff, 10. ** self.m_star_tot, 'y*', label=r"$M_{\ast, \rm TOT}$", markersize=15)
        plt.loglog(f.ar[w_max - 1] / f.r_eff, 5. * self.s_eff ** 2 * self.Re * self.kpc_arcsec / G, 'bs',
                   label=r"$5 \,\sigma_{\rm e}^2 R_{\rm e} / G$", markersize=12)
        plt.xlim([1e-2, 11])
        plt.xlabel(r"$r/R_{\rm e}$", fontsize=16)
        plt.ylabel(r"$M(r)/M_\odot$", fontsize=16)
        plt.legend(loc='best')
        if save_fig:
            plt.savefig('mass_prof_' + self.gal_name + '.pdf')
        plt.show()

    @staticmethod
    def _shift_systemic_velocity(V, xd):

        v_shift = (npmax(V[xd]) + npmin(V[xd])) / 2.
        print "shift:", v_shift
        return V - v_shift

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
                           save_fig=False, normalize=True, **kwargs):

        # plotting surface brightness profile with Sersic fits
        fig = plt.figure()
        ax = fig.add_axes((.1, .3, .8, .6))
        ax2 = fig.add_axes((.1, .1, .8, .2))

        R_arcsec, gc = self.rebin_profile(self.R_arcsec, self.gc, decimals=2)
        R_dat, SB_dat = self.get_surface_brightness(R_arcsec, gc)
        Re, n, I_0, Re_fix, n_fix, I_0_fix = self.fit_sersic_profile(R_dat, -SB_dat, Re_fix=Re_fix, **kwargs)

        if model is not None and isinstance(model, FJmodel):

            f = model

            if Re_model is None:
                print 'Projecting model to compute Re...'
                f.project(inclination=inclination, nx=60, scale='log', verbose=False)
                Re_model = f.r_eff

            r_mod, gc_mod = f.light_profile(inclination=inclination, nx=nx, npsi=nx / 2,
                                            Re_model=Re_model, Re_data=self.Re,
                                            xmin=self.R_arcsec[0], xmax=self.R_arcsec[-1], num=len(self.R_arcsec),
                                            **kwargs)

            gc_scale = self.gc[npabs(self.R_arcsec - self.Re).argmin()] - gc_mod[npabs(r_mod - self.Re).argmin()]

            # r_mod_new, gc_mod_new = self.rebin_profile(r_mod, gc_mod + gc_scale)
            R_mod, SB_mod = KinData.get_surface_brightness(r_mod, gc_mod + gc_scale)

            R_arcsec, gc = self.rebin_profile(self.R_arcsec, self.gc, decimals=2)
            R_mod, SB_mod, SB_std = self.rebin_profile(R_mod, SB_mod, x_array=R_arcsec)

            # w = nonzero(1. / SB_mod)  # exclude first non-infinite point
            if normalize:
                if Re_fix is not None:
                    # ax.plot(R_mod[w[0][1]:] / Re_fix, 10. ** (-0.4 * (SB_mod[w[0][1]:] -
                    #                                                  SB_mod[npabs(R_mod - Re_fix).argmin()])), 'ro',
                    ax.plot(R_dat / Re_fix, 10. ** (-0.4 * (SB_dat -
                                                            SB_dat[npabs(R_dat - Re_fix).argmin()])), 'bo-',
                            label=r"$\rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$")
                    ax.plot(R_mod / Re_fix, 10. ** (-0.4 * (SB_mod -
                                                            SB_mod[npabs(R_mod - Re_fix).argmin()])), 'ro--',
                            label=r"$f(\bf J) \, {\rm model}$")
                else:
                    # ax.plot(R_mod[w[0][1]:] / self.Re, 10. ** (-0.4 * (SB_mod[w[0][1]:] -
                    #                                                   SB_mod[npabs(R_mod - self.Re).argmin()])), 'ro',
                    ax.plot(R_dat / Re, 10. ** (-0.4 * (SB_dat - SB_dat[npabs(R_dat - Re).argmin()])), 'bo-',
                            label=r"$\rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$")
                    ax.plot(R_mod / self.Re, 10. ** (-0.4 * (SB_mod -
                                                             SB_mod[npabs(R_mod - self.Re).argmin()])), 'ro--',
                            label=r"$f(\bf J) \, {\rm model}$")
            else:
                # ax.plot(R_mod[w[0][1]:], SB_mod[w[0][1]:], 'ro', label=r"$f(\bf J) \, {\rm model}$")
                ax.plot(R_dat, SB_dat, 'bo-', label=r"$\rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$")
                ax.plot(R_mod, SB_mod, 'ro--', label=r"$f(\bf J) \, {\rm model}$")

            ax.set_ylabel(r"$I(R)/I(R_{\rm e})$", fontsize=18)
            ax.set_xscale('log')
            ax.set_xticklabels([])
            if normalize:
                ax.set_yscale('log')
                plot_data = [10. ** (-0.4 * (SB_dat - SB_dat[npabs(R_dat - Re_fix).argmin()])),
                             10. ** (-0.4 * (SB_mod - SB_mod[npabs(R_mod - Re_fix).argmin()]))]
                ax.set_ylim([.7 * npmin((plot_data[0])[~isnan(plot_data[0])]),
                             1e3])

            # Plot of the growth curves
            # plt.figure()
            # plt.plot(r_mod, gc_mod + gc_scale, 'bo', self.R_arcsec, self.gc, 'ro')
            ax.legend(loc='best', fontsize=16)

            '''
                Residuals plot
            '''
            ax2.set_xscale('log')
            ax2.axhline(0., color='k', ls='--')

            if Re_fix is not None:
                SB_dat = 10. ** (-0.4 * (SB_dat - SB_dat[npabs(R_dat - Re_fix).argmin()]))
                SB_mod = 10. ** (-0.4 * (SB_mod - SB_mod[npabs(R_mod - Re_fix).argmin()]))
            else:
                SB_dat = 10. ** (-0.4 * (SB_dat - SB_dat[npabs(R_dat - Re).argmin()]))
                SB_mod = 10. ** (-0.4 * (SB_mod - SB_mod[npabs(R_mod - Re).argmin()]))
            w = npabs((SB_dat - SB_mod) / SB_dat) < 10.
            residuals = ((SB_dat - SB_mod) / SB_dat)[w]
            if Re_fix is not None:
                ax2.plot(R_dat[w] / Re_fix, residuals, 'ko-')
            else:
                ax2.plot(R_dat[w] / Re, residuals, 'ko-')
            ax2.set_xlabel(r"$R/R_{\rm e}$", fontsize=18)
            ax2.set_ylabel(r"$\rm residuals$", fontsize=18)
            # ax2.set_ylim([residuals.min() * 1.25, residuals.max() * 1.25])
            # ticks = round(linspace(residuals.min(), residuals.max(), num=8),
            #               decimals=1)
            # ax2.yaxis.set_ticks(ticks)
            if save_fig:
                plt.savefig(self.gal_name + "_SB_wmodel.pdf", bbox_inches='tight')
            plt.show()

    def get_sb_profile(self, Re_fix=None, show=True, normalize=True, **kwargs):

        R_arcsec, gc = self.rebin_profile(self.R_arcsec, self.gc, decimals=2)
        R, sb = self.get_surface_brightness(R_arcsec, gc)
        Re, n, I_0, Re_fix, n_fix, I_0_fix = self.fit_sersic_profile(R, -sb, Re_fix=Re_fix, **kwargs)

        if normalize:
            if Re_fix is not None:
                plt.plot(R / Re_fix, 10. ** (-0.4 * (sb - sb[npabs(R - Re_fix).argmin()])), 'bo-',
                         label=r"$\rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$")
                plt.plot(R / Re_fix, 10. ** (+0.4 * (KinData.sersic(R, Re_fix, n_fix, I_0_fix) -
                         KinData.sersic(Re_fix, Re_fix, n_fix, I_0_fix))), 'k-', lw=2, label=u'Sérsic, n=%2.1f' % n_fix)
            else:
                plt.plot(R / Re, 10. ** (-0.4 * (sb - sb[npabs(R - Re).argmin()])), 'bo-',
                         label=r"$\rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$")
                plt.plot(R / Re, 10. ** (+0.4 * (KinData.sersic(R, Re_fix, n_fix, I_0_fix) -
                         KinData.sersic(Re_fix, Re_fix, n_fix, I_0_fix))), 'k-', lw=2, label=u'Sérsic, n=%2.1f' % n)
        else:
            plt.gca().invert_yaxis()
            plt.plot(R, sb, 'bo-', label=r"$\rm " + self.gal_name[:3] + "\," + self.gal_name[3:] + "$")
            plt.plot(R, -KinData.sersic(R, Re_fix, n_fix, I_0_fix), 'k-', lw=2, label=u'Sérsic, n=%2.1f' % n_fix)

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

    def _get_vel_curve_idx(self, X, Y, s, bins, vel, full_output=False):

        # theta, x, y = self.get_major_axis(X[s], Y[s], vel[bins[s]])
        theta, x = radians(90. + self.pa), linspace(X[s].min(), X[s].max(), num=100)
        y = tan(radians(90. + self.pa)) * x

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
    def rebin_profile(x, y, decimals=3, x_array=None):
        """
        Rebin the input profile so that values that have the same y are accreted in only one bin.
        The x value of the bin will be the mean x value.
        :param x:
        :param y:
        :param decimals:
        :param x_array:
        :return:
        """
        x_out, y_out, y_list, y_std = [], [], [], []
        if x_array is not None:
            for k in range(len(x_array)):
                x_k, y_k = [], []
                for i in range(len(x)):
                    if not isinf(y[i]) and not isnan(y[i]):
                        if npabs(x_array - x[i]).argmin() == k:
                            y_k.append(y[i])
                            x_k.append(x[i])
                x_out.append(x_array[k])
                y_out.append(array(y_k).mean())
                y_std.append(array(y_k).std())

            return array(x_out), array(y_out), array(y_std)
        else:
            for i in range(len(y)):
                if around(y[i], decimals=decimals) in around(y_list, decimals=decimals):
                    pass
                else:
                    y_list.append(y[i])
                    w = y == y[i]
                    y_out.append(y[w].mean())
                    x_out.append(x[w].mean())

            return array(x_out), array(y_out)

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

    def _get_kpc_over_arcsec(self):

        kpc_arcsec = None
        # read CALIFA_Lorenzo.conf file
        for n_line in range(24, 34):
            line = getline(self.conf_kpcarcsec_file, n_line)

            if line.split()[1] == self.gal_name:
                kpc_arcsec = float(line.split()[13])

        return kpc_arcsec

    def _get_masses(self):

        m_star, m_dyn = None, None
        s_eff, m_schw= None, None
        # read califa4dfmodels.dat file
        for n_line in range(19, 22):
            line = getline(self.conf_mass_file, n_line)

            if line.split()[1] == self.gal_name:
                m_star, s_eff, m_dyn, m_schw = float(line.split()[4]), float(line.split()[6]), float(line.split()[9]), \
                    float(line.split()[12])

        return m_star, s_eff, m_dyn, m_schw

    def _get_effective_radius(self):

        Re = None
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

        line = getline(self.aperture_file, 6)
        self.pa, self.pa_err = float(line.split()[0]), float(line.split()[1])

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
