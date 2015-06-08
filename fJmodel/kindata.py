__author__ = 'lposti'

from fJmodel import FJmodel
from sauron import sauron
from numpy import genfromtxt, zeros, linspace, exp, log10, column_stack, round, empty, nan, rot90, \
    meshgrid, radians, ones_like, where, average, sqrt, gradient, power, dstack
from numpy.ma import masked_array
from linecache import getline
from math import sin, cos
from numpy import sin as numpy_sin
from numpy import cos as numpy_cos
from numpy import max as npmax
from numpy import min as npmin
from numpy import abs as npabs
from numpy import reshape as numpy_reshape
from scipy.spatial import distance, cKDTree
from scipy.interpolate import RectBivariateSpline
import matplotlib.pylab as plt


class KinData(object):

    def __init__(self, directory=None, mge_file=None, kin_data_file=None, aperture_file=None, bins_file=None):

        if directory is not None:
            self.mge_file = directory + "/MGE_rband.txt"
            self.kin_data_file = directory + "/kin_data.dat"
            self.aperture_file = directory + "/aperture.dat"
            self.bins_file = directory + "/bins.dat"

            if directory[-1] == '/':
                self.gal_name = directory[-8:-1]
            else:
                self.gal_name = directory[-7:]
            print "Galaxy:", self.gal_name

        elif mge_file is not None and kin_data_file is not None and \
                aperture_file is not None and bins_file is not None:
            self.mge_file = mge_file
            self.kin_data_file = kin_data_file
            self.aperture_file = aperture_file
            self.bins_file = bins_file

        else:
            raise ValueError("ERROR: either pass the directory where the MGE, kinematic"
                             "and aperture data are or pass the filenames directly.")

        self.angle = float(getline(self.aperture_file, 4).split()[0])

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
                              reverse_v_field=False, **kwargs):

        if isinstance(model, FJmodel):
            f = model
        elif isinstance(model, basestring):
            f = FJmodel(model)
        else:
            raise NotImplementedError(" -- ERROR: either pass an FJmodel instance or full qualified path")

        # get data
        vel, sig, vel_err, sig_err, X, Y, bins, s, dx, minx, miny, nx, ny, xt, yt =\
            self._get_kinematic_data(full_output=True)

        vel_image = self.display_pixels(X[s], Y[s], vel[bins[s]], pixelsize=dx)
        sig_image = self.display_pixels(X[s], Y[s], sig[bins[s]], pixelsize=dx)

        # get MGE data
        mge = self._get_mge(xt=xt, yt=yt, angle=self.angle)

        # get model
        Rmax = 20.  # f.ar[-1]
        x, y = f.project(inclination=inclination, nx=30, npsi=31, Rmax=Rmax)

        # maxgrid: max value of the observed grid. Used to rescale the model image
        maxgrid = max(max(npmax(xt), abs(npmin(xt))), max(npmax(yt), abs(npmin(yt))))

        sigma, velocity, density = RectBivariateSpline(x, y, f.slos), RectBivariateSpline(x, y, f.vlos),\
            RectBivariateSpline(x, y, f.dlos)
        sigma_model, velocity_model = zeros(nx * ny), zeros(nx * ny)
        sig_model, vel_model = zeros(npmax(bins[s]) + 1), zeros(npmax(bins[s]) + 1)
        density_model = zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                x_rotated = (cos(radians(self.angle)) * xt[i] - sin(radians(self.angle)) * yt[ny - 1 - j]) \
                    * Rmax / maxgrid
                y_rotated = (sin(radians(self.angle)) * xt[i] + cos(radians(self.angle)) * yt[ny - 1 - j]) \
                    * Rmax / maxgrid
                sigma_model[i * ny + j] = sigma.ev(x_rotated, y_rotated)
                velocity_model[i * ny + j] = velocity.ev(x_rotated, y_rotated)

                x_rotated = (cos(radians(self.angle)) * xt[i] - sin(radians(self.angle)) * yt[j]) \
                    * Rmax / maxgrid
                y_rotated = (sin(radians(self.angle)) * xt[i] + cos(radians(self.angle)) * yt[j]) \
                    * Rmax / maxgrid
                density_model[i, j] = density.ev(x_rotated, y_rotated)

        # normalize density to its maximum
        density_model -= npmax(density_model)

        bins2 = numpy_reshape(bins, (ny, nx))
        bins_img = rot90(rot90(rot90(bins2)))
        bins2 = bins_img.reshape(bins_img.shape[0] * bins_img.shape[1])

        for i in range(max(bins[s]) + 1):
            w = where(bins2 == i)
            sig_model[i] = average(sigma_model[w] / npmax(sigma_model))
            vel_model[i] = average(velocity_model[w] / npmax(velocity_model))

        vel_image_mod = self.display_pixels(X[s], Y[s], vel_model[bins[s]], pixelsize=dx)
        sig_image_mod = self.display_pixels(X[s], Y[s], sig_model[bins[s]], pixelsize=dx)

        # peaks of velocity moments, used for re-scaling the model
        data_scale = npmax(vel[bins[s]]), npmax(sig[bins[s]])
        model_scale = npmax(vel_model[bins[s]]), npmax(sig_model[bins[s]])

        # colour scales of the velocity and velocity dispersion plots
        vmin, vmax = npmin(vel[bins[s]]), npmax(vel[bins[s]])
        smin, smax = npmin(sig[bins[s]]), npmax(sig[bins[s]])

        data_contour_levels = linspace(float((log10(mge)).min()) * .6, 0, num=6)
        model_contour_levels = linspace(float(density_model.min()) * .75, 0, num=6)

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
                           extent=[X[s].min() - dx, X[s].max() + dx,
                                   Y[s].min() - dx, Y[s].max() + dx], **kwargs)

        image.set_clim(vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(image)
        colorbar.set_label(r'$v$ [km/s]')
        # add density contours
        ax.contour(xt, yt, log10(mge).T, colors='k', levels=data_contour_levels)

        if one_figure:
            ax2 = fig.add_subplot(222)
        else:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
        ax2.set_xlabel("RA [arcsec]")
        ax2.set_ylabel("DEC [arcsec]")
        image2 = plt.imshow(sig_image, cmap=sauron, interpolation='nearest',
                            extent=[X[s].min() - dx, X[s].max() + dx,
                                    Y[s].min() - dx, Y[s].max() + dx], **kwargs)

        image2.set_clim(vmin=smin, vmax=smax)
        if one_figure:
            colorbar = fig.colorbar(image2)
        else:
            colorbar = fig2.colorbar(image2)
        colorbar.set_label(r'$\sigma$ [km/s]')
        # add density contours
        ax2.contour(xt, yt, log10(mge).T, colors='k', levels=data_contour_levels)

        if one_figure:
            ax3 = fig.add_subplot(223)
        else:
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
        ax3.set_xlabel("RA [arcsec]")
        ax3.set_ylabel("DEC [arcsec]")
        image3 = plt.imshow(vel_image_mod / model_scale[0] * data_scale[0], cmap=sauron, interpolation='nearest',
                            extent=[X[s].min() - dx, X[s].max() + dx,
                                    Y[s].min() - dx, Y[s].max() + dx], **kwargs)

        image3.set_clim(vmin=vmin, vmax=vmax)
        if one_figure:
            colorbar = fig.colorbar(image3)
        else:
            colorbar = fig3.colorbar(image3)
        colorbar.set_label(r'$v$ [km/s]')
        # add density contours
        ax3.contour(xt, yt, density_model.T, colors='k', levels=model_contour_levels)

        if one_figure:
            ax4 = fig.add_subplot(224)
        else:
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(111)
        ax4.set_xlabel("RA [arcsec]")
        ax4.set_ylabel("DEC [arcsec]")
        image4 = plt.imshow(sig_image_mod / model_scale[1] * data_scale[1], cmap=sauron, interpolation='nearest',
                            extent=[X[s].min() - dx, X[s].max() + dx,
                                    Y[s].min() - dx, Y[s].max() + dx], **kwargs)
        image4.set_clim(vmin=smin, vmax=smax)
        if one_figure:
            colorbar = fig.colorbar(image4)
        else:
            colorbar = fig4.colorbar(image4)
        colorbar.set_label(r'$\sigma$ [km/s]')
        # add density contours
        ax4.contour(xt, yt, density_model.T, colors='k', levels=model_contour_levels)

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

        # get model
        Rmax = 20.  # f.ar[-1]
        x, y = f.project(inclination=inclination, nx=30, npsi=31, Rmax=Rmax)

        # peaks of velocity moments, used for re-scaling the model
        data_scale = npmax(vel[bins[s]]), npmax(sig[bins[s]])
        model_scale = npmax(f.vlos), npmax(f.slos)

        # plot position-velocity diagrams
        fig = plt.figure(figsize=(14, 7.5))
        ax = fig.add_subplot(121)
        ax.set_xlabel("semi-major axis [arcsec]", fontsize=18)
        ax.set_ylabel("velocity [km/s]", fontsize=18)
        ax.plot(x / npmax(x) * npmax(X_xd_pv), f.vlos[:, len(f.vlos) / 2] / model_scale[0] * data_scale[0], 'b-')
        ax.errorbar(X_xd_pv, (vel[bins[s]])[xd], yerr=(vel_err[bins[s]])[xd], fmt='o', color='r', label=self.gal_name)

        ax2 = fig.add_subplot(122)
        ax2.set_xlabel("semi-major axis [arcsec]", fontsize=18)
        ax2.set_ylabel("velocity dispersion [km/s]", fontsize=18)
        ax2.plot(x / npmax(x) * npmax(X_xd_pv), f.slos[:, len(f.slos) / 2] / model_scale[1] * data_scale[1], 'b-')
        ax2.errorbar(X_xd_pv, (sig[bins[s]])[xd], yerr=(sig_err[bins[s]])[xd], fmt='o', color='r', label=self.gal_name)

        if save_fig:
            ax.legend(loc='best')
            ax2.legend(loc='best')
            plt.savefig('vprof_mod_comp' + self.gal_name + '.eps', bbox_inches='tight')
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

        xmin, xmax, ymin, ymax = X[s].min() - dx, X[s].max() + dx, Y[s].min() - dx, Y[s].max() + dx
        # use max and min of velocity curve to determine kinematic axis
        id_max, id_min = vel[bins[s]].argmax(), vel[bins[s]].argmin()

        # plot velocity map with major axis line
        x = linspace(xmin, xmax, num=120)
        y = x * (((Y[s])[id_max] - (Y[s])[id_min]) / ((X[s])[id_max] - (X[s])[id_min])) * 1.2

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
        s = where(bins > 0)[0]

        return bins, s

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
