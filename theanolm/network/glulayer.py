#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.weightfunctions import get_submatrix
from theanolm.network.basiclayer import BasicLayer

class GLULayer(BasicLayer):
    """Gated Linear Unit Layer

    Implements a convolution layer with a gating mechanism.

    Y. N. Dauphin (2016)
    Language Modeling with Gated Convolutional Networks
    https://arxiv.org/abs/1612.08083
    """

    def __init__(self, layer_options, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(layer_options, *args, **kwargs)

        if 'kernel_size' in layer_options:
            kernel_size = int(layer_options['kernel_size'])
        else:
            kernel_size = 5
        logging.debug("  kernel_size=%d", kernel_size)

        assert len(self._input_layers) == 1
        input_size = self._input_layers[0].output_size
        input_depth = self._input_layers[0].output_depth
        output_size = self.output_size
        output_depth = self.output_depth

        # Make sure the user hasn't tried to change the number of connections.
        if input_size != output_size:
            raise ValueError("GLU layer size has to match the previous layer.")

        # convolution filters for the linear projection and its gate
        # The width of the kernel is fixed to the width of the input vector.
        filter_shape = (output_depth, input_depth, kernel_size, input_size)
        self._init_weight('linear/W', filter_shape, scale=0.01)
        self._init_weight('gate/W', filter_shape, scale=0.01)

        # biases for the linear projection and its gate
        self._init_bias('linear/b', output_depth)
        self._init_bias('gate/b', output_depth)

        self._filter_shape = filter_shape
        self._input_size = input_size
        self._input_depth = input_depth

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
        input_depth = self._input_depth

        # Permutate the dimensions for conv2d(), which expects
        # [sequences, channels, rows, columns].
        if self._input_depth > 1:
            layer_input = layer_input.dimshuffle(1, 3, 0, 2)
        else:
            layer_input = layer_input.dimshuffle(1, 'x', 0, 2)

        # Shift the input right by k/2 time steps, where k is the kernel size,
        # so that the output at any time step does not contain information from
        # future words.
        padding_size = self._filter_shape[2] // 2
        padding = tensor.zeros([
            num_sequences, input_depth, padding_size, input_size])
        layer_input = tensor.concatenate([padding, layer_input], axis=2)

        # Compute the linear projection and the gate pre-activation. Because of
        # the padding, there are now more time steps than we want. If the filter
        # width (data width) is not an odd number, there will be one extra data
        # dimension too.
        linear = self._tensor_convolution(layer_input, 'linear')
        linear = linear.dimshuffle(2, 0, 3, 1)
        linear = linear[:num_time_steps,:,:input_size,:]
        gate = self._tensor_convolution(layer_input, 'gate')
        gate = gate.dimshuffle(2, 0, 3, 1)
        gate = gate[:num_time_steps,:,:input_size,:]

        # Add biases and multiply each element by the gate activation.
        bias = self._params[self._param_path('linear/b')]
        linear += bias
        bias = self._params[self._param_path('gate/b')]
        gate += bias
        self.output = linear * tensor.nnet.sigmoid(gate)

        if self.output_depth == 1:
            self.output = self.output.reshape([num_time_steps,
                                               num_sequences,
                                               input_size])

    def _tensor_convolution(self, input_matrix, param_name):
        """Convolves ``input_matrix`` using filters.

        ``input_matrix`` and the result have the shape expected by ``conv2d()``:
        [sequences, channels (depth), time steps, data].

        :type input_matrix: TensorVariable
        :param input_matrix: one or more sequences (first dimension) and one or
                             more channels (second dimension), each containing
                             two-dimensional data, first data dimension being
                             the time steps

        :type param_name: str
        :param param_name: name of a parameter group that contains a filter
                           matrix

        :rtype: TensorVariable
        :returns: the input convolved with the filters
        """

        filters = self._params[self._param_path(param_name) + '/W']
        return tensor.nnet.conv2d(input_matrix,
                                    filters,
                                    input_shape=(None, 1, None, None),
                                    filter_shape=self._filter_shape,
                                    border_mode='half',
                                    filter_flip=False)
