Basic Usage
===========

``theanolm`` command recognizes several subcommands:

theanolm train
  Trains a neural network language model.

theanolm score
  Performs text scoring and perplexity computation using a neural network
  language model.

theanolm decode
  Decodes a word lattice using a neural network to compute the language model
  probabilities.

theanolm sample
  Generates sentences by sampling words from a neural network language model.

theanolm version
  Displays the version number and exits.

The complete list of command line options available for each subcommand can be
displayed with the ``--help`` argument, e.g.::

    theanolm train --help

Using GPUs
==========

Theano can automatically utilize NVIDIA GPUs for numeric computation. First you
need to have CUDA installed. A GPU device can be selected using
``$THEANO_FLAGS`` environment variable, or in ``.theanorc`` configuration file.
For details about configuring Theano, see `Theano manual
<http://deeplearning.net/software/theano/library/config.html>`_. The simplest
way to get started is to set ``$THEANO_FLAGS`` as follows::

    export THEANO_FLAGS=floatX=float32,device=gpu

The device *gpu* selects an available GPU device and enables Theano's old GPU
backend. Only 32-bit floating point precision is supported, which is a good idea
anyway to conserve memory. The new GpuArray backend supports using multiple GPUs
simultaneously and 64-bit floats (although you generally want to conserve memory
by using 32-bit floats).

GpuArray backend can be enabled by selecting one of the *cuda* devices, e.g.
``device=cuda0``. Before using it you have to install the `libgpuarray`_
library. Also, currently it requires `cuDNN`_ for all the necessary operations
to work, and cuDNN requires a graphics card with compute capability 3.0 or
higher. The backend is still under active development, so you should use the
latest developmet versions of Theano and libgpuarray from GitHub.

In order to use multiple GPUs, one would map the *cuda* devices to *dev* names,
e.g::

    export THEANO_FLAGS=floatX=float32,contexts=dev0->cuda0;dev1->cuda1"

.. _libgpuarray: http://deeplearning.net/software/libgpuarray/installation.html
.. _cuDNN: https://developer.nvidia.com/cudnn
