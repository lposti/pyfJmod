#!/usr/bin/env python

__author__ = 'lposti'

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

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
