#######################################################
#
#   f(J) models plotting utilities
#   uses Matplotlib and numpy
#
#######################################################

__author__ = 'lposti'


from numpy import log10
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
        self.nplots = nrow*ncols
        self.idplot = -1
        self.fig, self.ax = plt.subplots(nrow, ncols, sharex=sharex, sharey=sharey, **fig_kw)

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

    def plotFigure(self, name=None):

        if self.nplots == 1:
            if self.xlabel is not None:
                self.ax.set_xlabel(self.xlabel, fontsize=self.fontsize)
            if self.ylabel is not None:
                self.ax.set_ylabel(self.ylabel, fontsize=self.fontsize)
        else:
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
        if self.idplot >= 0:
            if name is not None:
                plt.savefig(name, bbox_inches='tight')
            else:
                plt.show()
        else:
            raise ValueError("No Plot to show!!")


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

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = 0

        self.loglog(R, self.fJ.rho(R, z))
        if show:
            self.plotFigure()

    def plotSigR(self, R=None, z=None, show=True):

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = 0

        self.plot(log10(R), self.fJ.sigR(R, z))
        if show:
            self.plotFigure()

    def plotSigz(self, R=None, z=None, show=True):

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = 0

        self.plot(log10(R), self.fJ.sigz(R, z))
        if show:
            self.plotFigure()

    def plotSigp(self, R=None, z=None, show=True):

        if R is None:
            R = self.fJ.ar
        if z is None:
            z = 0

        self.plot(log10(R), self.fJ.sigp(R, z))
        if show:
            self.plotFigure()