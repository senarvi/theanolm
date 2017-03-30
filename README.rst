TheanoLM
========

Introduction
------------

TheanoLM is a recurrent neural network language modeling tool implemented using
the Python library `Theano <http://www.deeplearning.net/software/theano/>`_.
Theano allows the user to customize and extend the neural network very
conveniently, still generating highly efficient code that can utilize multiple
GPUs or CPUs for parallel computation. TheanoLM allows the user to specify an
arbitrary network architecture. New layer types and optimization methods can be
easily implemented.

TheanoLM can be used for rescoring n-best lists, decoding word lattices, or
generating text. It can be called from command line or from a Python script.

Implementations of some of the currently popular layer types such as long
short-term memory (LSTM) and gated recurrent units (GRU), their bidirectional
versions, and highway networks are provided. Several different Stochastic
Gradient Descent (SGD) based optimizers are implemented, including RMSProp,
AdaGrad, ADADELTA, and Adam. In addition to the standard cross-entropy cost, one
can use sampling based noise-contrastive estimation or BlackOut.

About the project
-----------------

TheanoLM is open source and licensed under the `Apache License, Version 2.0
<LICENSE.txt>`__. The source code is available on `GitHub
<https://github.com/senarvi/theanolm>`_. Documentation can be read online on
`Read the Docs <http://theanolm.readthedocs.io/en/latest/>`_.

Author
------

| Seppo Enarvi
| http://senarvi.github.io/
