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
<http://deeplearning.net/software/theano/library/config.html>`_. Currently
Theano supports only 32-bit floating point precision, when using a GPU. The
simplest way to get started is to set ``$THEANO_FLAGS`` as follows::

    export THEANO_FLAGS=floatX=float32,device=gpu
