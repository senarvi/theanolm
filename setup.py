#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This setup script can be used to run unit tests, manually install the
package, and upload the package to PyPI.

python3 setup.py --help       - Display help.
python3 setup.py test         - Execute unit tests.
python3 setup.py install      - Install the package.
python3 setup.py sdist upload - Upload the project to PyPI.
"""

from os import path
from setuptools import setup, find_packages

SCRIPT_DIR = path.dirname(path.realpath(__file__))
VERSION_PATH = path.join(SCRIPT_DIR, 'theanolm', 'version.py')

# Don't import theanolm, as the user may not have the dependencies installed
# yet. This will import __version__.
with open(VERSION_PATH, 'r') as version_file:
    exec(version_file.read())
VERSION = __version__ #@UndefinedVariable

LONG_DESCRIPTION = 'TheanoLM is a recurrent neural network language modeling ' \
                   'toolkit implemented using Theano. Theano allows the user ' \
                   'to customize and extend the neural network very ' \
                   'conveniently, still generating highly efficient code ' \
                   'that can utilize multiple GPUs or CPUs for parallel ' \
                   'computation. TheanoLM allows the user to specify ' \
                   'arbitrary network architecture. New layer types and ' \
                   'optimization methods can be easily implemented.'
KEYWORDS = 'theano neural network language modeling machine learning research'
CLASSIFIERS = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Programming Language :: Python :: 3',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: Apache Software License',
               'Operating System :: OS Independent',
               'Topic :: Scientific/Engineering']

setup(name='TheanoLM',
      version=VERSION,
      author='Seppo Enarvi',
      author_email='seppo2017@marjaniemi.com',
      url='https://github.com/senarvi/theanolm',
      download_url='https://github.com/senarvi/theanolm/tarball/v' + VERSION,
      description='Toolkit for neural network language modeling using Theano',
      long_description=LONG_DESCRIPTION,
      license='Apache License, Version 2.0',
      keywords=KEYWORDS,
      classifiers=CLASSIFIERS,
      packages=find_packages(exclude=['tests']),
      package_data={'theanolm': ['architectures/*.arch']},
      scripts=['bin/theanolm', 'bin/wctool'],
      install_requires=['numpy', 'Theano', 'h5py'],
      test_suite='tests')
