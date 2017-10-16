TheanoLM
========

Introduction
------------

TheanoLM is a recurrent neural network language modeling tool implemented using
the Python library `Theano`_. Theano allows the user to customize and extend the
neural network very conveniently, still generating highly efficient code that
can utilize multiple GPUs or CPUs for parallel computation. TheanoLM allows the
user to specify an arbitrary network architecture. New layer types and
optimization methods can be easily implemented.

TheanoLM can be used for rescoring n-best lists and Kaldi lattices, decoding HTK
word lattices, and generating text. It can be called from command line or from a
Python script.

Implementations of many currently popular layer types are provided, such as
`long short-term memory (LSTM)`_, `gated recurrent units (GRU)`_, `bidirectional
recurrent networks`_, `gated linear units`_, and `highway networks`_ are
provided. Several different Stochastic Gradient Descent (SGD) based optimizers
are implemented, including `RMSProp`_, `AdaGrad`_, `ADADELTA`_, and `Adam`_.

There are several features that are especially useful with very large
vocabularies. The effective vocabulary size can be reduced by using a class
model. TheanoLM supports also subword vocabularies create e.g. using
`Morfessor`_. In addition to the standard cross-entropy cost, one can use
sampling based `noise-contrastive estimation (NCE)`_  or `BlackOut`_.

.. _Theano: http://www.deeplearning.net/software/theano/
.. _long short-term memory (LSTM): https://www.researchgate.net/publication/13853244_Long_Short-term_Memory
.. _gated recurrent units (GRU): https://arxiv.org/abs/1406.1078
.. _bidirectional recurrent networks: http://ieeexplore.ieee.org/document/650093/
.. _gated linear units (GLU): https://arxiv.org/abs/1612.08083
.. _highway networks: https://arxiv.org/abs/1505.00387
.. _RMSProp: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
.. _AdaGrad: http://jmlr.org/papers/v12/duchi11a.html
.. _ADADELTA: https://arxiv.org/abs/1212.5701
.. _Adam: https://arxiv.org/abs/1412.6980
.. _Morfessor: https://github.com/aalto-speech/morfessor
.. _noise-contrastive estimation (NCE): http://www.jmlr.org/papers/v13/gutmann12a.html
.. _BlackOut: https://arxiv.org/abs/1511.06909

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
