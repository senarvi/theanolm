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

Using a GPU
-----------

Theano can automatically utilize NVIDIA GPUs for numeric computation. First you
need to have CUDA installed. A GPU device can be selected using
``$THEANO_FLAGS`` environment variable, or in ``.theanorc`` configuration file.
For details about configuring Theano, see `Theano manual
<http://deeplearning.net/software/theano/library/config.html>`_. The simplest
way to get started is to set ``$THEANO_FLAGS`` as follows::

    export THEANO_FLAGS=floatX=float32,device=gpu

The device *gpu* selects an available GPU device and enables Theano's old GPU
backend. Only 32-bit floating point precision is supported, which is a good idea
anyway to conserve memory. The new GpuArray backend can be enabled by setting
one of the *cuda0*, *cuda1*, ... devices. Before using it you have to install
the `libgpuarray`_ library.

.. _libgpuarray: http://deeplearning.net/software/libgpuarray/installation.html
