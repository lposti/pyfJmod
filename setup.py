#!/usr/bin/env python

__author__ = 'lposti'

from distutils.core import setup

setup(name='pyfJmod',
      version='0.2.1',
      description='Python package for f(J) models analysis',
      author='Lorenzo Posti',
      author_email='lorenzo.posti@gmail.com',
      packages=['fJmodel'],
      url='https://github.com/lposti/pyfJmod',
      package_data={'fJmodel': ['examples/*']},
      requires=['numpy', 'matplotlib', 'progressbar'],
      # install_requires=['numpy', 'matplotlib'],
      classifiers=['Development Status :: 2 - Pre-Alpha',
                   'Environment :: Console',
                   'Environment :: MacOS X',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX :: Linux',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Utilities'],
      )
