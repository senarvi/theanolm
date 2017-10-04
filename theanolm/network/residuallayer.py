#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the fully-connected tanh layer.
"""

import theano.tensor as tensor

from theanolm.network.basiclayer import BasicLayer

class ResidualLayer(BasicLayer):
    """Layer with Hyperbolic Tangent Activation

    A layer that uses hyperbolic tangent activation function.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters for this layer.
        """

        super().__init__(*args, **kwargs)

        # Create the parameters. Weight matrix and bias for concatenated input.
        input_size = self._input_layers[0].output_size
        output_size = self.output_size
        for input_layer in self._input_layers:
            if input_layer.output_size != input_size:
               raise ValueError("All inputs of a residual layer have to be equal size.")
        self._init_weight('input/W', (input_size, output_size), scale=0.01)
        self._init_bias('input/b', output_size)

        self.output = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer.
        """

        #layer_input = tensor.concatenate([x.output for x in self._input_layers],
                                         #axis=2)

        #preact = self._tensor_preact(layer_input, 'input')
        preact = self._tensor_preact(self._input_layers[0].output, 'input')
        for input_layer in self._input_layers:
            preact = tensor.add(preact, input_layer.output)
        self.output = self._activation(preact)
