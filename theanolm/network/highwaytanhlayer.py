#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.weightfunctions import get_submatrix
from theanolm.network.basiclayer import BasicLayer

class HighwayTanhLayer(BasicLayer):
    """Highway Network Layer with Hyperbolic Tangent Activation

    R. K. Srivastava (2015)
    Highway Networks
    ICML 2015 Deep Learning Workshop
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters for this layer.
        """

        super().__init__(*args, **kwargs)

        # Make sure the user hasn't tried to change the number of connections.
        input_size = sum(x.output_size for x in self.input_layers)
        output_size = self.output_size
        if input_size != output_size:
            raise ValueError("Highway network layer cannot change the number "
                             "of connections.")

        # Create the parameters. Normal weight matrix and bias are concatenated
        # with those of the transform gate. Transform gate bias is initialized
        # to a negative value, so that the network is initially biased towards
        # carrying the input without transformation.
        self._init_weight('input/W', (input_size, output_size), scale=0.01,
                          count=2)
        self._init_bias('input/b', output_size, [0.0, -1.0])

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer.
        """

        layer_input = tensor.concatenate([x.output for x in self.input_layers],
                                         axis=2)
        preact = self._tensor_preact(layer_input, 'input')
        # normal activation (hidden state) and transform gate
        h = tensor.tanh(get_submatrix(preact, 0, self.output_size))
        t = tensor.nnet.sigmoid(get_submatrix(preact, 1, self.output_size))
        self.output = h * t + layer_input * (1 - t)
