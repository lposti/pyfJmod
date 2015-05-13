# pyfJmod
Python analysis tool for f(J) models

[![Build Status](https://travis-ci.org/lposti/pyfJmod.svg?branch=master)](https://travis-ci.org/lposti/pyfJmod)
[![Coverage Status](https://coveralls.io/repos/lposti/pyfJmod/badge.svg?branch=master)](https://coveralls.io/r/lposti/pyfJmod?branch=master)
[![PyPI](https://img.shields.io/pypi/v/pyfJmod.svg)](https://pypi.python.org/pypi/pyfJmod)
[![License](https://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/lposti/pyfJmod/blob/master/LICENSE.md)

## Installation

### Prerequisites

- `numpy` Python package
- `scipy` Python package
- `matplotlib` Python package
- `progressbar` Python package

I suggest to use `pip` (install in Ubuntu as:`sudo apt-get install python-pip`) by typing 

```
sudo pip install numpy
sudo pip install scipy
sudo pip install matplotlib
sudo pip install progressbar
```

### Clone the package (*recommended*)

Clone the repository in your preferred directory:
```
git clone https://github.com/lposti/pyfJmod
```

Then `cd` to the cloned directory and install the package:

```
sudo python setup.py install
```


### Use `pip` (*alternative*)

Get `pip`, if you don't have it:

```
sudo apt-get install python-pip
```

Then use `pip` to install the package with all its dependencies:

```
sudo pip install pyfJmod
``` 

### Tests

Install `nose` unittesting package:

```
sudo pip install nose
```

Then `cd` to the cloned `pyfJmod` directory and type:

```
nosetests
```
Please notice that it'll take a few minutes to complete.


## Examples

The following example will show how to plot a density profile (along the major axis) for an f(**J**) model:

```python
from fJmodel.fJmodel import *
from fJmodel.fJplots import *

# load the f(J) model data
f = FJmodel("/full/path/to/file/fJmodel_file.out")

# create the matplotlib interface to directly plot the data
p0 = FJmodelPlot(f, xlabel=r"$\log\, r$", ylabel=r"$\rho$", fontsize=18)

# make & show the density profile plot
p0.plotRho()
```

The following example will show how to plot the radial, vertical and azimuthal velocity dispersion profiles (along the major axis) for an f(**J**) model:

```python
from fJmodel.fJmodel import *
from fJmodel.fJplots import *

# load the f(J) model data
f = FJmodel("/full/path/to/file/fJmodel_file.out")

# create the matplotlib interface to directly plot the data
p1 = FJmodelPlot(f, xlabel=r"$\log\, r$", ylabel=r"$\sigma$", fontsize=18)

# make the sigma profiles, show only on the last one
p1.plotSigmaR(show=False)
p1.plotSigmaz(show=False)
p1.plotSigmap()
```
