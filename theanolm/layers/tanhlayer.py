#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.layers.basiclayer import BasicLayer

class TanhLayer(BasicLayer):
    """Hyperbolic Tangent Layer for Neural Network Language Model

    A layer that uses hyperbolic tangent activation activation function. If
    multiple inputs are specified, combines them by addition.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters for this layer.
        """

        super().__init__(*args, **kwargs)

        # Create the parameters. Weight matrix and bias for each input.
        output_size = self.output_size
        for input_index, input_layer in enumerate(self.input_layers):
            input_size = input_layer.output_size
            param_name = 'input' + str(input_index) + '/W'
            self._init_random_weight(param_name, input_size, output_size, scale=0.01)
            param_name = 'input' + str(input_index) + '/b'
            self._init_bias(param_name, output_size)

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        preacts = []
        for input_index, input_layer in enumerate(self.input_layers):
            param_name = 'input' + str(input_index)
            preacts.append(self._tensor_preact(input_layer.output, param_name))
        self.output = tensor.tanh(sum(preacts))
