__author__ = 'lp1osti'

from fJmodel import FJmodel
import numpy as np
import matplotlib.pylab as plt


def hernq(x, m):
    x = np.asarray(x)
    return m/x/pow(.5+x, 3)


def nfw(x, m):
    x = np.asarray(x)
    return m/x/pow(.5+x, 2)


def plot():
    f = FJmodel("/Users/lp1osti/git_fJmodels/models/Hernq_0.55_0.55_1.00_1.00_4.out")
    x = np.logspace(-2.5, 1.5)
    plt.loglog(x, f.rho(x, 0), 'ro', x, hernq(x, .1), 'k-')
    plt.show()


def plot2comp():
    f = FJmodel("/Users/lp1osti/git_fJmodels/models/Hernq_0.55_0.55_1.00_1.00_4.out")
    g = FJmodel("/Users/lp1osti/git_fJmodels/models/NFW_0.55_0.55_1.00_1.00_4.out")
    x = np.logspace(-2.5, 1.5)
    plt.loglog(x, f.rho(x, 0), 'ro', x, hernq(x, .1), 'k-')
    plt.loglog(x, g.rho(x, 0), 'bo', x, nfw(x, .1), 'k--')
    plt.show()

if __name__ == "__main__":
    plot2comp()