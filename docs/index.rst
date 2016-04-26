TheanoLM
========

TheanoLM is a recurrent neural network language modeling tool implemented using
the Python library `Theano <http://www.deeplearning.net/software/theano/>`_.
Theano allows the user to customize and extend the neural network very
conveniently, still generating highly efficient code that can utilize multiple
GPUs or CPUs for parallel computation. TheanoLM allows the user to specify an
arbitrary network architecture. New layer types and optimization methods can be
easily implemented. Implementations of common layer types, such as long
short-term memory and gated recurrent units, and Stochastic Gradient Descent,
RMSProp, AdaGrad, ADADELTA, and Adam optimizers are provided.

Getting Started section of this guide provides an introduction how to get
started with training neural network language models and performing various
operations with them. The development documentation is intended to help
extending the toolkit.

Getting Started
---------------

.. toctree::
   :maxdepth: 2

   installation
   usage
   training
   applying

Development
-----------

.. toctree::
   :maxdepth: 2

   contributing
   modules

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
