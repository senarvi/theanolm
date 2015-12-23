#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.layers.basiclayer import BasicLayer

class DropoutLayer(BasicLayer):
    """Dropout Layer for Neural Network Language Model

    A dropout layer is not a regular layer in the sense that it doesn't contain
    any neurons. It simply randomly sets some activations to zero at train time
    to prevent overfitting.    

    N. Srivastava et al. (2014)
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    Journal of Machine Learning Research, 15, 1929-1958
    """

    def __init__(self, *args, **kwargs):
        """Validates the parameters of this layer.
        """

        super().__init__(*args, **kwargs)

        # Make sure the user hasn't tried to change the number of connections.
        input_size = self.input_layers[0].output_size
        output_size = self.output_size
        if input_size != output_size:
            raise ValueError("Dropout layer cannot change the number of connections.")

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        layer_input = self.input_layers[0].output
        self.output = tensor.switch(
            self.network.is_training,
            layer_input * self.network.random.binomial(
                layer_input.shape, p=0.5, n=1, dtype=layer_input.dtype),
            layer_input * 0.5)
