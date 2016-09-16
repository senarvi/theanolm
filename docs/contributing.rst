Contributing
============

You're welcome to contribute.

1. Fork the repository on GitHub.
2. Clone the forked repository into a local directory:
   ``git clone my-repository-url``
3. Create a new branch: ``git checkout -b my-new-feature``
4. Commit your changes: ``git commit -a``
5. Push to the branch: ``git push origin my-new-feature``
6. Submit a pull request on GitHub.

Structure of the source code
----------------------------

``theanolm.commands`` package contains the main scripts for launching the
subcommands.

``theanolm.network.Network`` class stores the neural network state. It is
constructed from layers that are implemented in ``theanolm.layers`` package.
Each layer implements functions for constructing the symbolic layer structure.

``theanolm.trainers`` package contains classes that perform the training
iterations. They are responsible for cross-validation and learning rate
adjustment. They use one of the optimizers found in ``theanolm.optimizers``
package to perform the actual parameter update.

``theanolm.textscorer.TextScorer`` class is used to score text, both for
cross-validation during training and by the score command for evaluating text.
``theanolm.textsampler.TextSampler`` class is used by the sample command for
generating text.

Layers
------

Layers receive a list of input layers. Every layer has ``create_structure()``
method that describes its output, given the output of its input layers. The
output is a 3-dimensional tensor variable, a symbolic representation of the
neural network data at that point of the computation graph. It is important to
realize that the actual data can have any values and any shape. The first
dimension is the time step and the second dimension is the sequence of the
mini-batch. The third dimension is the actual layer output.
