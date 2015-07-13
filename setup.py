#!/usr/bin/env python

__author__ = 'lposti'

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
from sys import argv
from subprocess import Popen

# Make a `veryclean` rule to get rid of intermediate and library files
if "veryclean" in argv[1:]:
    print "Deleting cython files..."
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    Popen("rm -rf build", shell=True, executable="/bin/bash")
    Popen("rm -rf *.c", shell=True, executable="/bin/bash")
    Popen("rm -rf *.so", shell=True, executable="/bin/bash")

    # Now do a normal clean
    argv[1] = "clean"

setup(name='pyfJmod',
      version='0.2.2',
      description='Python package for f(J) models analysis',
      author='Lorenzo Posti',
      author_email='lorenzo.posti@gmail.com',
      packages=['fJmodel'],
      url='https://github.com/lposti/pyfJmod',
      package_data={'fJmodel': ['examples/*']},
      requires=['scipy', 'numpy', 'matplotlib', 'progressbar', 'cython'],
      # install_requires=['numpy', 'matplotlib'],
      classifiers=['Development Status :: 2 - Pre-Alpha',
                   'Environment :: Console',
                   'Environment :: MacOS X',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: POSIX :: Linux',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Utilities'],
      ext_modules=[Extension('projection_cy', ['fJmodel/projection_cy.pyx'], include_dirs=[get_include()])],
      cmdclass={'build_ext': build_ext}
      )
