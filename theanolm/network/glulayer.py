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
        self._input_depth = input_depth

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is 3- or 4-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third and fourth
        dimension are the layer input. The fourth dimension is created when more
        than one filter is used.
        """

        if not self._network.mode.minibatch:
            raise RuntimeError("Text generation is not possible with "
                               "convolution layers.")

        layer_input = self._input_layers[0].output
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]

        # Compute the linear projection and the gate pre-activation.
        linear = self._tensor_convolution(layer_input, 'linear')
        gate = self._tensor_convolution(layer_input, 'gate')
        self.output = linear * tensor.nnet.sigmoid(gate)

    def _tensor_convolution(self, input_matrix, param_name):
        """Convolves ``input_matrix`` using filters and adds a bias.

        ``input_matrix`` and the result normally have the shape of a mini-batch:
        the first dimension is the time step and the second dimension is the
        sequence. The last dimension is always the data vector. The size of the
        input data vector should equal to the first dimension of the weight
        vector, and the second dimension of the weight vector defines the size
        of the output data vector.

        :type input_matrix: TensorVariable
        :param input_matrix: the preactivations will be computed by multiplying
                             the data vectors (the last dimension of this
                             matrix) by the weight matrix, and adding bias

        :type param_name: str
        :param param_name: name of a parameter group that contains a filter
                           matrix and a bias vector

        :rtype: TensorVariable
        :returns: a matrix that has the same number of dimensions as
                  ``input_matrix``, but the data vectors (the last dimension of
                  this matrix) are the preactivations
        """

        num_time_steps = input_matrix.shape[0]
        num_sequences = input_matrix.shape[1]
        output_size = input_matrix.shape[2]

        # Permutate the dimensions for conv2d(), which expects
        # [sequences, channels, rows, columns].
        if self._input_depth > 1:
            input_matrix = input_matrix.dimshuffle(1, 3, 0, 2)
        else:
            input_matrix = input_matrix.dimshuffle(1, 'x', 0, 2)

        filters = self._params[self._param_path(param_name) + '/W']
        bias = self._params[self._param_path(param_name) + '/b']

        result = tensor.nnet.conv2d(input_matrix,
                                    filters,
                                    input_shape=(None, 1, None, None),
                                    filter_shape=self._filter_shape,
                                    border_mode='half',
                                    filter_flip=False)
        result = result.dimshuffle(2, 0, 3, 1)
        result = result[:num_time_steps,:,:output_size,:]
        result += bias

        if self.output_depth == 1:
            result = result.reshape([num_time_steps,
                                     num_sequences,
                                     output_size])

        return result
