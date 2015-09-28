#######################################################
#
#   f(J) models plotting utilities
#   uses Matplotlib and numpy
#
#######################################################

__author__ = 'lposti'


from numpy import log10, meshgrid, linspace, zeros, reshape, arccos, dot, degrees, cos, pi, sqrt
from numpy import max as npmax
from numpy import min as npmin
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from fJmodel import FJmodel
from voronoi import displayBinnedMap


class PlotInterface(object):

    def __init__(self, xlabel=None, ylabel=None, fontsize=15,
                 nrow=1, ncols=1, sharex=False, sharey=False, **fig_kw):
        """
        Constructor of the class
        :param xlabel: label for x axis (str, raw str)
        :param ylabel: label for y axis (str, raw str)
        :param fontsize: fontsize for x-ylabels
        :param nrow: number of subplots in row (int)
        :param ncols: number of subplots in column (int)
        :param sharex: whether subplots must share x axis (bool)
        :param sharey: whether subplots must share x axis (bool)
        :param fig_kw: dictionary passed to plt.figure
        :return: Initializes variables fig (plt.figure) and ax (list of plt.axes)
                 and nplots, idplot
        """
        self.xlabel = None
        if xlabel is not None:
            self.xlabel = xlabel

        self.ylabel = None
        if ylabel is not None:
            self.ylabel = ylabel

        self.fontsize = fontsize
        self.nplots = nrow * ncols
        self.idplot = -1
        self.nrow, self.ncols = nrow, ncols
        self.sharex, self.sharey = sharex, sharey
        self.fig_kw = fig_kw

        # init figure and axes
        self.fig, self.ax = self._init_fig(**self.fig_kw)

    def _init_fig(self, **fig_kw):
        """
        Private method: initialize Figure and Axes instances for the class.
        Used both at the beginning (constructor) and at the end (after plotFigure)
        :return: handles to Figure and Axes instances
        """
        fig, ax = plt.subplots(self.nrow, self.ncols, sharex=self.sharex, sharey=self.sharey, **fig_kw)
        return fig, ax

    def plot(self, xdata, ydata, samefig=False, **kwargs):

        if not samefig or self.idplot < 0:
            self.idplot += 1

        if self.nplots == 1:
            return self.ax.plot(xdata, ydata, **kwargs)
        else:
            return self.ax[self.idplot].plot(xdata, ydata, **kwargs)

    def loglog(self, xdata, ydata, samefig=False, **kwargs):

        if not samefig or self.idplot < 0:
            self.idplot += 1

        if self.nplots == 1:
            return self.ax.loglog(xdata, ydata, **kwargs)
        else:
            return self.ax[self.idplot].loglog(xdata, ydata, **kwargs)

    def text(self, x, y, text, samefig=False, **kwargs):

        if not samefig or self.idplot < 0:
            self.idplot += 1

        if self.nplots == 1:
            self.ax.text(x, y, text, **kwargs)
        else:
            self.ax[self.idplot].text(x, y, text, **kwargs)

    def contour(self, x, y, z, samefig=False, **kwargs):

        if not samefig or self.idplot < 0:
            self.idplot += 1

        # here I transpose the rho matrix...
        # is there any other fix with this matplotlib issue?
        if self.nplots == 1:
            return self.ax.contour(x, y, z.T, **kwargs)
        else:
            return self.ax[self.idplot].contour(x, y, z.T, **kwargs)

    def contourf(self, x, y, z, samefig=False, **kwargs):

        if not samefig or self.idplot < 0:
            self.idplot += 1

        # here I transpose the rho matrix...
        # is there any other fix with this matplotlib issue?
        if self.nplots == 1:
            contour = self.ax.contourf(x, y, z.T, **kwargs)
            plt.colorbar(contour, ax=self.ax)
            return contour
        else:
            contour = self.ax[self.idplot].contourf(x, y, z.T, **kwargs)
            plt.colorbar(contour, ax=self.ax[self.idplot])
            return contour

    def imshow(self, f, xmin, xmax, ymin, ymax, num=100, samefig=False, **kwargs):

        # check if f is callable
        assert hasattr(f, '__call__')

        if not samefig or self.idplot < 0:
            self.idplot += 1

        m = zeros((num, num))
        x = linspace(xmin, xmax, num=num)
        y = linspace(ymin, ymax, num=num)

        for i in range(num):
            for j in range(num):
                m[i, j] = f(x[i], y[j])

        # here I transpose the rho matrix...
        # is there any other fix with this matplotlib issue?
        if self.nplots == 1:
            image = self.ax.imshow(m.T, extent=[xmin, xmax, ymin, ymax], origin='lower', **kwargs)
            plt.colorbar(image, ax=self.ax)
            return image
        else:
            image = self.ax[self.idplot].imshow(m.T, extent=[xmin, xmax, ymin, ymax], origin='lower', **kwargs)
            plt.colorbar(image, ax=self.ax[self.idplot])
            return image

    def displayBinMap(self, samefig=False, **kwargs):

        if not samefig or self.idplot < 0:
            self.idplot += 1

        if self.nplots == 1:
            img = displayBinnedMap(subax=self.ax, **kwargs)
            plt.colorbar(img, ax=self.ax)
            return img
        else:
            img = displayBinnedMap(subax=self.ax[self.idplot], **kwargs)
            plt.colorbar(img, ax=self.ax[self.idplot])
            return img

    def plotEllipse(self, xy, width, height, angle, samefig=False):

        if not samefig or self.idplot < 0:
            self.idplot += 1

        ell = Ellipse(xy=xy, width=width, height=height, angle=angle)

        if self.nplots == 1:
            self.ax.add_artist(ell)
            ell.set_edgecolor('k')
            ell.set_linewidth(.75)
            ell.set_facecolor('none')
        else:
            raise NotImplementedError('How many plots?!')

    def plotFigure(self, name=None, legend=False):

        if self.nplots == 1:

            # x-y labels if condition
            if self.xlabel is not None:
                self.ax.set_xlabel(self.xlabel, fontsize=self.fontsize)
            if self.ylabel is not None:
                self.ax.set_ylabel(self.ylabel, fontsize=self.fontsize)

            # Legend if condition
            if legend:
                self.ax.legend(loc='best')

        else:
            # x-y labels if condition
            if type(self.xlabel) is list:
                for i in range(self.nplots):
                    if self.xlabel[i] is not None:
                        self.ax[i].set_xlabel(self.xlabel[i], fontsize=self.fontsize)
            else:
                for i in range(self.nplots):
                    if self.xlabel is not None:
                        self.ax[i].set_xlabel(self.xlabel, fontsize=self.fontsize)

            if type(self.ylabel) is list:
                for i in range(self.nplots):
                    if self.ylabel[i] is not None:
                        self.ax[i].set_ylabel(self.ylabel[i], fontsize=self.fontsize)
            else:
                for i in range(self.nplots):
                    if self.ylabel is not None:
                        self.ax[i].set_ylabel(self.ylabel, fontsize=self.fontsize)

            # Legend if condition
            if legend:
                for i in range(self.nplots):
                    self.ax[i].legend(loc='best')

        if self.idplot >= 0:
            if name is not None:
                plt.savefig(name, bbox_inches='tight')
            else:
                plt.show()
        else:
            raise ValueError("No Plot to show!!")

        # now reset the fig. and axes to the init values
        self.fig, self.ax = self._init_fig(**self.fig_kw)


class FJmodelPlot(PlotInterface):
    """
    Class for handling plots directly via f(J) model's data

    Inherited from class PlotInterface: the constructor
    explicitly calls that of PlotInterface
    """
    def __init__(self, fJ, xlabel=None, ylabel=None, fontsize=15,
                 nrow=1, ncols=1, sharex=False, sharey=False, **fig_kw):
        """
        Constructor of the class. Inherits properties from PlotInterface
        explicitly calling its constructor
        :param fJ: instance of FJmodel class (checked by assertion)
        :param xlabel: label for x axis (str, raw str)
        :param ylabel: label for y axis (str, raw str)
        :param nrow: number of subplots in row (int)
        :param ncols: number of subplots in column (int)
        :param sharex: whether subplots must share x axis (bool)
        :param sharey: whether subplots must share x axis (bool)
        :param fig_kw: dictionary passed to plt.figure
        :return: Initializes fJ and PlotInterface
        """
        assert type(fJ) is FJmodel
        self.fJ = fJ
        PlotInterface.__init__(self, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize,
                               nrow=nrow, ncols=ncols, sharex=sharex, sharey=sharey, **fig_kw)

    def plotRho(self, R=None, z=None, show=True):

        self._pltloglog(self.fJ.rho, R, z, show)

    def plotSigR(self, R=None, z=None, show=True):

        self._pltsemilog(self.fJ.sigR, R, z, show)

    def plotSigz(self, R=None, z=None, show=True):

        self._pltsemilog(self.fJ.sigz, R, z, show)

    def plotSigp(self, R=None, z=None, show=True):

        self._pltsemilog(self.fJ.sigp, R, z, show)

    def plotPhi(self, R=None, z=None, show=True):

        self._pltloglog(lambda x, y: -self.fJ.phi(x, y), R, z, show)

    def _pltloglog(self, f, R=None, z=None, show=True):

        # check if f is callable
        assert hasattr(f, '__call__')

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = 0

        self.loglog(R, f(R, z))
        if show:
            self.plotFigure()

    def _pltsemilog(self, f, R=None, z=None, show=True):

        # check if f is callable
        assert hasattr(f, '__call__')

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = 0

        self.plot(log10(R), f(R, z))
        if show:
            self.plotFigure()

    def contourfRho(self, R=None, z=None, show=True):

        self._logcontourf(self.fJ.rho, R, z, show)

    def contourfVrot(self, R=None, z=None, show=True):

        self._pltcontourf(self.fJ.vrot, R, z, show)

    def imshowRho(self, Rmin=None, Rmax=None, zmin=None, zmax=None, show=True):

        self._logimshow(self.fJ.rho, Rmin, Rmax, zmin, zmax, show)

    def imshowVrot(self, Rmin=None, Rmax=None, zmin=None, zmax=None, show=True):

        self._pltimshow(self.fJ.vrot, Rmin, Rmax, zmin, zmax, show)

    def contourRho(self, R=None, z=None, show=True, **kwargs):

        self._logcontour(self.fJ.rho, R, z, show, **kwargs)

    def contourVrot(self, R=None, z=None, show=True, **kwargs):

        self._pltcontour(self.fJ.vrot, R, z, show, **kwargs)

    def displayRhoBinnedMap(self, show=True):

        X, Y = self.fJ.voronoiBin()
        self._displayBinnedMap(self.fJ.dlos, X, Y)
        if show:
            self.plotFigure()

    def displaySigBinnedMap(self, show=True):

        self.contourRho(R=linspace(-npmax(self.fJ.ar), npmax(self.fJ.ar)),
                        z=linspace(-npmax(self.fJ.ar), npmax(self.fJ.ar)), show=False, colors='w', linewidths=2)

        X, Y = self.fJ.voronoiBin()
        self._displayBinnedMap(self.fJ.slos, X, Y)
        if show:
            self.plotFigure()

    def displayVrotBinnedMap(self, show=True):

        self.contourRho(R=linspace(-npmax(self.fJ.ar), npmax(self.fJ.ar)),
                        z=linspace(-npmax(self.fJ.ar), npmax(self.fJ.ar)), show=False, colors='k', linewidths=2)

        X, Y = self.fJ.voronoiBin()
        self._displayBinnedMap(self.fJ.vlos, X, Y)
        if show:
            self.plotFigure()

    def plotProjectedRhoContour(self, inclination=90., show=True, **kwargs):

        self._plot_projection(field='density', inclination=inclination, show=show, **kwargs)

    def plotProjectedSigmaContour(self, inclination=90., show=True, **kwargs):

        self._plot_projection(field='sigma', inclination=inclination, show=show, **kwargs)

    def plotProjectedVelocityContour(self, inclination=90., show=True, **kwargs):

        self._plot_projection(field='velocity', inclination=inclination, show=show, **kwargs)

    def plotProjectedRhoProfile(self, inclination=90., show=True, **kwargs):

        self._plot_projected_profile(field='density', inclination=inclination, show=show, **kwargs)

    def plotProjectedSigmaProfile(self, inclination=90., show=True, **kwargs):

        self._plot_projected_profile(field='sigma', inclination=inclination, show=show, **kwargs)

    def plotProjectedVelocityProfile(self, inclination=90., show=True, **kwargs):

        self._plot_projected_profile(field='velocity', inclination=inclination, show=show, **kwargs)

    def plotVelocityEllipsoids(self, rad=30., num=10, rmin=None, rmax=None, text=None, show=True):

        X0, Y0, X1, Y1, X2, Y2, X3, Y3 = self.fJ.velocity_ellipsoids(v_len=rad / 2, num=num, rmin=rmin, rmax=rmax)
        x0, y0, w0, w1, v00, v01, v10, v11 = self.fJ.velocity_ellipsoids(v_len=rad / 2, num=num, rmin=rmin,
                                                                         rmax=rmax, plot_ellipses=True)

        if rmin is None:
            rmin = self.fJ.r_half
        if rmax is None:
            rmax = 20. * self.fJ.r_half

        rmin, rmax, rad = rmin / self.fJ.r_half, rmax / self.fJ.r_half, rad / self.fJ.r_half
        X0, Y0 = X0 / self.fJ.r_half, Y0 / self.fJ.r_half
        x0, y0 = x0 / self.fJ.r_half, y0 / self.fJ.r_half

        X1, Y1, X2, Y2, X3, Y3 = X1 / self.fJ.r_half, Y1 / self.fJ.r_half, X2 / self.fJ.r_half, Y2 / self.fJ.r_half, \
            X3 / self.fJ.r_half, Y3 / self.fJ.r_half

        self.ax.set_xlim([-rmin * 0.4, rmax * 1.05])
        self.ax.set_ylim([-rmin * 0.4, rmax * 1.05])
        self.ax.set_aspect('equal')

        if text is not None:
            self.text(1.75, 2.75, text, samefig=True, fontsize=18)

        ci = cos(linspace(0., pi / 2))
        N_angles = num
        N = len(x0)

        for i in range(N_angles, N, N_angles):
            self.plot(y0[i] * ci, y0[i] * sqrt(1. - ci * ci), color='k', ls='--', lw=.75)

        i, step, flag = N_angles, 1, False
        while i < N:
            if flag:
                if i % N_angles < step:
                    step = 1
                    if rad * w0[i] * N_angles > 1. * y0[i]:
                        step += 1

            width, height = rad * w0[i], rad * w1[i]
            angle = degrees(arccos(dot([v00[i], v01[i]], [1, 0])))
            if angle > 135.:
                angle = -angle
            self.plot(xdata=[X0[i], X1[i]], ydata=[Y0[i], Y1[i]], samefig=True, color='k', lw=2)
            self.plot(xdata=[X2[i], X3[i]], ydata=[Y2[i], Y3[i]], samefig=True, color='r', lw=2)
            self.plotEllipse((x0[i], y0[i]), width=width, height=height, angle=angle, samefig=True)
            i += step

        if show:
            self.plotFigure()

    def _plot_projected_profile(self, field='density', inclination=90., show=True, **kwargs):

        x, y = self.fJ.project(inclination=inclination, nx=60, npsi=31, scale='log')

        if field is 'density':
            self.plot(log10(x[len(x) / 2:]), self.fJ.dlos[len(x) / 2:, len(y) / 2], **kwargs)
        elif field is 'sigma':
            self.plot(x[len(x) / 2:], self.fJ.slos[len(x) / 2:, len(y) / 2], **kwargs)
        elif field is 'velocity':
            self.plot(x[:], self.fJ.vlos[:, len(y) / 2], **kwargs)

        if show:
            self.plotFigure()

    def _plot_projection(self, field='density', inclination=90., show=True, **kwargs):

        x, y = self.fJ.project(inclination=inclination, nx=60, npsi=31)

        if field is 'density':
            self.contourf(x, y, self.fJ.dlos, **kwargs)
        elif field is 'sigma':
            self.contourf(x, y, self.fJ.slos, **kwargs)
        elif field is 'velocity':
            self.contourf(x, y, self.fJ.vlos, **kwargs)

        if show:
            self.plotFigure()

    def _logcontourf(self, f, R=None, z=None, show=True, **kwargs):

        # check if f is callable
        assert hasattr(f, '__call__')

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = self.fJ.ar

        X, Y = meshgrid(R, z)

        self.contourf(X, Y, log10(f(R, z)), **kwargs)
        if show:
            self.plotFigure()

    def _pltcontourf(self, f, R=None, z=None, show=True, **kwargs):

        # check if f is callable
        assert hasattr(f, '__call__')

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = self.fJ.ar

        X, Y = meshgrid(R, z)

        self.contourf(X, Y, f(R, z), **kwargs)
        if show:
            self.plotFigure()

    def _logcontour(self, f, R=None, z=None, show=True, **kwargs):

        # check if f is callable
        assert hasattr(f, '__call__')

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = self.fJ.ar

        X, Y = meshgrid(R, z)

        self.contour(X, Y, log10(f(R, z)), **kwargs)
        if show:
            self.plotFigure()

    def _pltcontour(self, f, R=None, z=None, show=True, **kwargs):

        # check if f is callable
        assert hasattr(f, '__call__')

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = self.fJ.ar

        X, Y = meshgrid(R, z)

        self.contour(X, Y, f(R, z), **kwargs)
        if show:
            self.plotFigure()

    def _logimshow(self, f, Rmin=None, Rmax=None, zmin=None, zmax=None, show=True):

        # check if f is callable
        assert hasattr(f, '__call__')

        if Rmin is None:
            Rmin = self.fJ.ar[0]
        if Rmax is None:
            Rmax = self.fJ.ar[-1]
        if zmin is None:
            zmin = self.fJ.ar[0]
        if zmax is None:
            zmax = self.fJ.ar[-1]

        self.imshow(lambda x, y: log10(f(x, y)), Rmin, Rmax, zmin, zmax)
        if show:
            self.plotFigure()

    def _pltimshow(self, f, Rmin=None, Rmax=None, zmin=None, zmax=None, show=True):

        # check if f is callable
        assert hasattr(f, '__call__')

        if Rmin is None:
            Rmin = self.fJ.ar[0]
        if Rmax is None:
            Rmax = self.fJ.ar[-1]
        if zmin is None:
            zmin = self.fJ.ar[0]
        if zmax is None:
            zmax = self.fJ.ar[-1]

        self.imshow(lambda x, y: f(x, y), Rmin, Rmax, zmin, zmax)
        if show:
            self.plotFigure()

    def _displayBinnedMap(self, quantity_map, X, Y):

        self.displayBinMap(binNum=self.fJ.binNum, signal=reshape(quantity_map, len(X)), x_sig=X, y_sig=Y)
