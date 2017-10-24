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

Theano can automatically utilize NVIDIA GPUs for numeric computation. Whether
the CPU or a GPU is used, is selected by configuring Theano. This is totally
transparent to TheanoLM.

First you need to have CUDA installed. The new GpuArray backend is the only GPU
backend that Theano supports anymore. Before using it you have to install the
`libgpuarray`_ library. Also, currently it requires `cuDNN`_ for all the
necessary operations to work, and cuDNN requires a graphics card with compute
capability 3.0 or higher. The backend is still under active development, so
using the latest developmet versions of Theano and libgpuarray from GitHub is
recommended.

The first GPU device can be selected using ``device=cuda0`` in ``$THEANO_FLAGS``
environment variable, or in ``.theanorc`` configuration file. The simplest way
to get started is to set ``$THEANO_FLAGS`` as follows::

    export THEANO_FLAGS=floatX=float32,device=cuda0

``floatX=float32`` selects 32-bit floating point precision, which is not
required anymore in the new backend, but is a good idea to conserve memory. In
order to use multiple GPUs, one would map the *cuda* devices to *dev* names,
e.g::

    export THEANO_FLAGS=floatX=float32,contexts=dev0->cuda0;dev1->cuda1"

For details on configuring Theano, see `Theano Configuration`_ in the API
documentation.

.. _libgpuarray: http://deeplearning.net/software/libgpuarray/installation.html
.. _cuDNN: https://developer.nvidia.com/cudnn
.. _Theano Configuration: http://deeplearning.net/software/theano/library/config.html
