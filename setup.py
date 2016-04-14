#!/usr/bin/env python3

import os
import subprocess
from glob import glob
from setuptools import setup, find_packages

script_dir = os.path.dirname(os.path.realpath(__file__))
version = subprocess.check_output(['git', 'describe'], cwd=script_dir)
version = version.decode('utf-8').rstrip()[1:]
scripts = glob(os.path.join(script_dir, 'bin', '*'))

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
      version=version,
      author='Seppo Enarvi',
      author_email='seppo2016@marjaniemi.com',
      url='https://github.com/senarvi/theanolm',
      description='Toolkit for neural network language modeling using Theano',
      long_description=long_description,
      license='Apache License, Version 2.0',
      keywords=keywords,
      classifiers=classifiers,
      packages=find_packages(),
      package_data={'theanolm': ['architectures/*.arch']},
      scripts=scripts,
      install_requires=['numpy', 'scipy', 'theano', 'h5py'])
