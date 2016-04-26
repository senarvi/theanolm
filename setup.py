#!/usr/bin/env python3

import sys
from os import path
import subprocess
from glob import glob
import re
from setuptools import setup, find_packages

script_dir = path.dirname(path.realpath(__file__))
version_path = path.join(script_dir, 'theanolm', 'version.py')
scripts = glob(path.join(script_dir, 'bin', '*'))

# Don't import theanolm, as the user may not have the dependencies installed
# yet. This will import __version__.
with open(version_path, 'r') as version_file:
    exec(version_file.read())

long_description = 'TheanoLM is a recurrent neural network language modeling ' \
                   'toolkit implemented using Theano. Theano allows the user ' \
                   'to customize and extend the neural network very ' \
                   'conveniently, still generating highly efficient code ' \
                   'that can utilize multiple GPUs or CPUs for parallel ' \
                   'computation. TheanoLM allows the user to specify ' \
                   'arbitrary network architecture. New layer types and ' \
                   'optimization methods can be easily implemented.'
keywords = 'theano neural network language modeling machine learning research'
classifiers = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Programming Language :: Python :: 3',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: Apache Software License',
               'Operating System :: OS Independent',
               'Topic :: Scientific/Engineering']

setup(name='TheanoLM',
      version=__version__,
      author='Seppo Enarvi',
      author_email='seppo2016@marjaniemi.com',
      url='https://github.com/senarvi/theanolm',
      download_url='https://github.com/senarvi/theanolm/tarball/v' + __version__,
      description='Toolkit for neural network language modeling using Theano',
      long_description=long_description,
      license='Apache License, Version 2.0',
      keywords=keywords,
      classifiers=classifiers,
      packages=find_packages(exclude=['tests']),
      package_data={'theanolm': ['architectures/*.arch']},
      scripts=['bin/theanolm', 'bin/wctool'],
      install_requires=['numpy', 'scipy', 'Theano', 'h5py'],
      test_suite='tests')
