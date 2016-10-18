Introduction
============

TheanoLM is a recurrent neural network language modeling tool implemented using
the Python library `Theano <http://www.deeplearning.net/software/theano/>`_.
Theano allows the user to customize and extend the neural network very
conveniently, still generating highly efficient code that can utilize multiple
GPUs or CPUs for parallel computation. TheanoLM allows the user to specify an
arbitrary network architecture. New layer types and optimization methods can be
easily implemented. Implementations of common layer types, such as long
short-term memory and gated recurrent units, and Stochastic Gradient Descent,
RMSProp, AdaGrad, ADADELTA, and Adam optimizers are provided.

Publications
============

Seppo Enarvi and Mikko Kurimo (2016), `TheanoLM — An Extensible Toolkit for
Neural Network Language Modeling <https://arxiv.org/abs/1605.00942>`_. In
Proceedings of the 17th Annual Conference of the International Speech
Communication Association (INTERSPEECH).

License
=======

TheanoLM is licensed under the `Apache License, Version 2.0
<https://github.com/senarvi/theanolm/blob/master/LICENSE.txt>`_.
