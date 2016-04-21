#!/usr/bin/env python3

import sys
from os import path
import subprocess
from glob import glob
import re
from setuptools import setup, find_packages

script_dir = path.dirname(path.realpath(__file__))
scripts = glob(path.join(script_dir, 'bin', '*'))
pkginfo_path = path.join(script_dir, 'PKG-INFO')

try:
    tag = subprocess.check_output(['git', 'describe', '--match', 'vv[0-9]*'],
                                  cwd=script_dir,
                                  stderr=subprocess.STDOUT)
    tag = tag.decode('utf-8').rstrip()
    version = tag[1:]
except:
    version = None
if version is None:
    if not path.exists(pkginfo_path):
        print("setup.py can only be run from a Git repository or from a "
              "distribution that includes distutils metadata (PKG-INFO).")
        sys.exit(1)
    version_re = re.compile(r'^Version: +(\d.*)')
    with open(pkginfo_path, 'r') as pkginfo_file:
        for line in pkginfo_file:
            match = version_re.search(line)
            if match:
                version = match.group(1).strip()
                break
if version is None:
    print("Version was not found from Git repository or PKG-INFO.")
    sys.exit(1)

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
      download_url='https://github.com/senarvi/theanolm/tarball/v' + version,
      description='Toolkit for neural network language modeling using Theano',
      long_description=long_description,
      license='Apache License, Version 2.0',
      keywords=keywords,
      classifiers=classifiers,
      packages=find_packages(),
      package_data={'theanolm': ['architectures/*.arch']},
      scripts=scripts,
      install_requires=['numpy', 'scipy', 'theano', 'h5py'],
      test_suite='tests')
