#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Gated Linear Unit layer.
"""

import logging

import theano.tensor as tensor

from theanolm.network.basiclayer import BasicLayer
from theanolm.network.weightfunctions import get_submatrix

class GLULayer(BasicLayer):
    """Gated Linear Unit Layer

    Implements a convolution layer with a gating mechanism.

    Y. N. Dauphin (2017)
    Language Modeling with Gated Convolutional Networks
    Proc. International Conference on Machine Learning
    """

    def __init__(self, layer_options, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(layer_options, *args, **kwargs)

        if 'filter_size' in layer_options:
            filter_size = int(layer_options['filter_size'])
        else:
            filter_size = 5
        logging.debug("  filter_size=%d", filter_size)

        assert len(self._input_layers) == 1
        input_size = self._input_layers[0].output_size
        output_size = self.output_size

        # concatenation of convolution filters for the linear projection and its
        # gate
        filter_shape = (filter_size, input_size, output_size)
        self._init_weight('input/W', filter_shape, scale=0.01, count=2)

        # concatenation of biases for the linear projection and its gate
        self._init_bias('input/b', output_size * 2)

        self._filter_size = filter_size
        self._input_size = input_size

        self.output = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is 3- or 4-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third and fourth
        dimension are the layer input. The fourth dimension is created when more
        than one filter is used.
        """

        if not self._network.mode.minibatch:
            raise RuntimeError("Text generation and lattice decoding are not "
                               "possible with convolution layers.")

        layer_input = self._input_layers[0].output
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        input_size = self._input_size

        # Shift the input right by k - 1 time steps, where k is the filter size,
        # so that the output at any time step does not contain information from
        # future words.
        padding_size = self._filter_size - 1
        padding = tensor.zeros([padding_size, num_sequences, input_size])
        layer_input = tensor.concatenate([padding, layer_input])

        # Compute the linear projection and the gate pre-activation in a single
        # convolution operation.
        preact = self._tensor_conv1d(layer_input, 'input')
        linear = get_submatrix(preact, 0, self.output_size)
        gate = get_submatrix(preact, 1, self.output_size)

        self.output = linear * tensor.nnet.sigmoid(gate)
