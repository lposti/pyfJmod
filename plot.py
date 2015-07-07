__author__ = 'lp1osti'

from fJmodel.fJmodel import FJmodel
from fJmodel.kindata import KinData
from fJmodel.sauron import sauron
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


def plot3galaxies():

    # get the data

    k1 = KinData('/Users/lp1osti/Dropbox/fJ_CALIFA/data/NGC2592')
    k2 = KinData('/Users/lp1osti/Dropbox/fJ_CALIFA/data/NGC6125')
    k3 = KinData('/Users/lp1osti/Dropbox/fJ_CALIFA/data/NGC6427')
    ks = [k1, k2, k3]

    f1 = FJmodel('fJmodel/examples/Hernq_flat_dp0.75_dz4_c5_4.out')
    f2 = FJmodel('fJmodel/examples/Hernq_flat_dp0.75_dz4_c5_4.out')
    f3 = FJmodel('fJmodel/examples/Hernq_flat_dp0.75_dz4_c5_4.out')
    fs = [f1, f2, f3]

    incls, Rs = [90., 40., 90.], [8., 8., 8.]

    fig = plt.figure(figsize=(4., 6), dpi=80)
    gs = [gridspec.GridSpec(2, 3), gridspec.GridSpec(2, 3)]
    gs[0].update(top=0.95, bottom=0.525, left=0.02, right=0.98, wspace=0., hspace=0.)
    gs[1].update(top=0.475, bottom=0.05, left=0.02, right=0.98, wspace=0., hspace=0.)

    for i, k in enumerate([k1, k2, k3]):

        vel, sig, vel_err, sig_err, X, Y, bins, s, dx, minx, miny, nx, ny, xt, yt =\
            k._get_kinematic_data(full_output=True)

        vel_image = k.display_pixels(X[s], Y[s], vel[bins[s]], pixelsize=dx)
        sig_image = k.display_pixels(X[s], Y[s], sig[bins[s]], pixelsize=dx)

        # get MGE data
        mge = k._get_mge(xt=xt, yt=yt, angle=k.angle)

        # get model data
        vel_model, sig_model, density_model = k._get_model_kinematics(fs[i], incls[i], Rs[i],
                                                                         s, bins, xt, yt, nx, ny)

        vel_image_mod = k.display_pixels(X[s], Y[s], vel_model[bins[s]], pixelsize=dx)
        sig_image_mod = k.display_pixels(X[s], Y[s], sig_model[bins[s]], pixelsize=dx)

        data_scale = max(np.max(vel[bins[s]]), np.max(sig[bins[s]])),\
            max(np.max(vel[bins[s]]), np.max(sig[bins[s]]))
        model_scale = max(np.max(vel_model[bins[s]]), np.max(sig_model[bins[s]])),\
            max(np.max(vel_model[bins[s]]), np.max(sig_model[bins[s]]))

        # colour scales of the velocity and velocity dispersion plots
        vmin, vmax = np.min(vel[bins[s]]), np.max(vel[bins[s]])
        smin, smax = np.min(sig[bins[s]]), np.max(sig[bins[s]])

        vm_min, vm_max = np.min(vel_model[bins[s]] / model_scale[0] * data_scale[0]),\
            np.max(vel_model[bins[s]] / model_scale[0] * data_scale[0])
        sm_min, sm_max = np.min(sig_model[bins[s]] / model_scale[1] * data_scale[1]),\
            np.max(sig_model[bins[s]] / model_scale[1] * data_scale[1])

        data_contour_levels = np.linspace(float((np.log10(mge)).min()) * 0.6, 0, num=6)
        model_contour_levels = np.linspace(float(density_model.min()) * 0.6, 0, num=6)

        #
        # Begin plotting loop
        #

        ax = plt.subplot(gs[0][0, i])
        v_img = plt.imshow(vel_image, cmap=sauron, interpolation='nearest',
                           extent=[X[s].min() - dx, X[s].max() + dx,
                                   Y[s].min() - dx, Y[s].max() + dx])
        v_img.set_clim(vmin=vmin, vmax=vmax)
        # add density contours
        ax.contour(xt, yt, np.log10(mge).T, colors='k', levels=data_contour_levels)
        ax.text(0.45, 0.9, "[%4.0f, %4.0f]" % (vmin, vmax), transform=ax.transAxes, fontsize=8)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        ax2 = plt.subplot(gs[0][1, i])
        vm_img = plt.imshow(vel_image_mod / model_scale[0] * data_scale[0], cmap=sauron, interpolation='nearest',
                            extent=[X[s].min() - dx, X[s].max() + dx,
                                    Y[s].min() - dx, Y[s].max() + dx])
        vm_img.set_clim(vmin=vmin, vmax=vmax)
        # add density contours
        ax2.contour(xt, yt, density_model.T, colors='k', levels=model_contour_levels)
        ax2.text(0.45, 0.9, "[%4.0f, %4.0f]" % (vm_min, vm_max), transform=ax2.transAxes, fontsize=8)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        ax3 = plt.subplot(gs[1][0, i])
        s_img = plt.imshow(sig_image, cmap=sauron, interpolation='nearest',
                           extent=[X[s].min() - dx, X[s].max() + dx,
                                   Y[s].min() - dx, Y[s].max() + dx])
        s_img.set_clim(vmin=smin, vmax=smax)
        # add density contours
        ax3.contour(xt, yt, np.log10(mge).T, colors='k', levels=data_contour_levels)
        ax3.text(0.45, 0.9, "[%4.0f, %4.0f]" % (smin, smax), transform=ax3.transAxes, fontsize=8)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)

        ax4 = plt.subplot(gs[1][1, i])
        sm_img = plt.imshow(sig_image_mod / model_scale[1] * data_scale[1], cmap=sauron, interpolation='nearest',
                            extent=[X[s].min() - dx, X[s].max() + dx,
                                    Y[s].min() - dx, Y[s].max() + dx])
        sm_img.set_clim(vmin=smin, vmax=smax)
        # add density contours
        ax4.contour(xt, yt, density_model.T, colors='k', levels=model_contour_levels)
        ax4.text(0.45, 0.9, "[%4.0f, %4.0f]" % (sm_min, sm_max), transform=ax4.transAxes, fontsize=8)
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)

    plt.savefig("/Users/lp1osti/Desktop/stocaz.pdf")
    plt.show()



if __name__ == "__main__":

    plot3galaxies()
