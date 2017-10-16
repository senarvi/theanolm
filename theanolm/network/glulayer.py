#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Gated Linear Unit layer.
"""

import logging

import theano.tensor as tensor

from theanolm.backend import conv1d
from theanolm.network.basiclayer import BasicLayer

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

        # convolution filters for the linear projection and its gate
        # The width of the filter is fixed to the width of the input vector.
        filter_shape = (filter_size, input_size, output_size)
        self._init_weight('linear/W', filter_shape, scale=0.01)
        self._init_weight('gate/W', filter_shape, scale=0.01)

        # biases for the linear projection and its gate
        self._init_bias('linear/b', output_size)
        self._init_bias('gate/b', output_size)

        self._filter_shape = filter_shape
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

        # Shift the input right by k/2 time steps, where k is the filter size,
        # so that the output at any time step does not contain information from
        # future words.
        padding_size = self._filter_shape[2] // 2
        padding = tensor.zeros([padding_size, num_sequences, input_size])
        layer_input = tensor.concatenate([padding, layer_input])

        # Compute the linear projection and the gate pre-activation. Because of
        # the padding, there are now more time steps than we want.
        linear = self._tensor_conv1d(layer_input, 'linear')
        linear = linear[:num_time_steps]
        gate = self._tensor_conv1d(layer_input, 'gate')
        gate = gate[:num_time_steps]

        # Add biases and multiply each element by the gate activation.
        bias = self._params[self._param_path('linear/b')]
        linear += bias
        bias = self._params[self._param_path('gate/b')]
        gate += bias
        self.output = linear * tensor.nnet.sigmoid(gate)

    def _tensor_conv1d(self, input_matrix, param_name):
        """Convolves ``input_matrix`` using filters.

        :type input_matrix: symbolic 3D tensor
        :param input_matrix: one or more sequences of features in the shape
                             (time steps, sequences, features)

        :type param_name: str
        :param param_name: name of a parameter group that contains a filter
                           matrix

        :rtype: symbolic 3D tensor
        :returns: the input convolved with the filters in the shape (time steps,
                  sequences, features)
        """

        # Permutate input dimensions from (time steps, sequences, features) to
        # (samples, elements, features).
        input_matrix = input_matrix.dimshuffle(1, 0, 2)

        filters = self._params[self._param_path(param_name) + '/W']
        result = conv1d(input_matrix,
                        filters,
                        padding='same')

        # Permutate input dimensions from (samples, elements, features) to
        # (time steps, sequences, features).
        return result.dimshuffle(1, 0, 2)
