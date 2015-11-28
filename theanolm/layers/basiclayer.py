#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import random_weight, orthogonal_weight

class BasicLayer(object):
    """Superclass for Neural Network Layers
    """

    def __init__(self, layer_name, input_layers, output_size, is_recurrent=False):
        """Saves some attributes that are common to all layers.
        """

        self.name = layer_name
        self.input_layers = input_layers
        self.output_size = output_size
        self.is_recurrent = is_recurrent

        logging.debug("- %s name=%s inputs=[%s] size=%d",
            self.__class__.__name__,
            layer_name,
            ', '.join([x.name for x in input_layers]),
            self.output_size)

        self.param_init_values = OrderedDict()

    def set_params(self, params):
        self._params = params

    def _get_param(self, param_name):
        return self._params[self.name + '.' + param_name]

    def _init_random_weight(self, param_name, input_size, output_size, scale=None, count=1):
        """ Generates a weight matrix from “standard normal” distribution.

        :type input_size: int
        :param input_size: size of the input dimension of the weight

        :type output_size: int
        :param output_size: size of the output dimension of the weight

        :type scale: float
        :param scale: if other than None, the matrix will be scaled by this factor

        :rtype: numpy.ndarray
        :returns: the generated weight matrix
        """

        self.param_init_values[self.name + '.' + param_name] = \
            numpy.concatenate([random_weight(input_size, output_size, scale=0.01)
                               for _ in range(count)],
                              axis=1)

    def _init_orthogonal_weight(self, param_name, input_size, output_size, scale=None, count=1):
        """ Generates a weight matrix from “standard normal” distribution. If
        in_size matches out_size, generates an orthogonal matrix.

        :type input_size: int
        :param input_size: size of the input dimension of the weight

        :type output_size: int
        :param output_size: size of the output dimension of the weight

        :type scale: float
        :param scale: if other than None, the matrix will be scaled by this factor,
                      unless an orthogonal matrix is created
        """

        self.param_init_values[self.name + '.' + param_name] = \
            numpy.concatenate([orthogonal_weight(input_size, output_size, scale=0.01)
                               for _ in range(count)],
                              axis=1)

    def _init_zero_bias(self, param_name, size):
        self.param_init_values[self.name + '.' + param_name] = \
            numpy.zeros((size,)).astype(theano.config.floatX)

    def _tensor_preact(self, input_matrix, param_name):
        weight = self._params[self.name + '.' + param_name + '.W']
        bias = self._params[self.name + '.' + param_name + '.b']
        return tensor.dot(input_matrix, weight) + bias
