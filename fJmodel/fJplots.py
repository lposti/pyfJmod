#######################################################
#
#   f(J) models plotting utilities
#   uses Matplotlib and numpy
#
#######################################################

__author__ = 'lposti'


from numpy import log10, meshgrid, linspace, zeros
import matplotlib.pylab as plt
from fJmodel import FJmodel


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

    def _logcontourf(self, f, R=None, z=None, show=True):

        # check if f is callable
        assert hasattr(f, '__call__')

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = self.fJ.ar

        X, Y = meshgrid(R, z)

        self.contourf(X, Y, log10(f(R, z)))
        if show:
            self.plotFigure()

    def _pltcontourf(self, f, R=None, z=None, show=True):

        # check if f is callable
        assert hasattr(f, '__call__')

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = self.fJ.ar

        X, Y = meshgrid(R, z)

        self.contourf(X, Y, f(R, z))
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