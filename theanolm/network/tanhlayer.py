#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.basiclayer import BasicLayer

class TanhLayer(BasicLayer):
    """Hyperbolic Tangent Activation Layer

    A layer that uses hyperbolic tangent activation activation function. If
    multiple inputs are specified, combines them by addition.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters for this layer.
        """

        super().__init__(*args, **kwargs)

        # Create the parameters. Weight matrix and bias for concatenated input.
        input_size = sum(x.output_size for x in self.input_layers)
        output_size = self.output_size
        self._init_random_weight('input/W',
                                 (input_size, output_size),
                                 scale=0.01)
        self._init_bias('input/b', output_size)

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        layer_input = tensor.concatenate([x.output for x in self.input_layers],
                                         axis=2)
        preact = self._tensor_preact(layer_input, 'input')
        self.output = tensor.tanh(preact)
