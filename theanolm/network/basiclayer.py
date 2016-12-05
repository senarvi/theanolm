#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.weightfunctions import random_weight, orthogonal_weight

class BasicLayer(object, metaclass=ABCMeta):
    """Superclass for Neural Network Layers
    """

    def __init__(self, layer_options, network, profile=False):
        """Saves some attributes that are common to all layers.

        :type layer_options: dict
        :param layer_options: dictionary of layer options

        :type network: Network
        :param network: the network object creating this layer

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.name = layer_options['name']
        self.input_layers = layer_options['input_layers']
        self._devices = layer_options['devices']

        if 'size' in layer_options:
            self.output_size = int(layer_options['size'])
        else:
            self.output_size = \
                sum([x.output_size for x in self.input_layers])

        num_devices = len(self._devices)
        self._weight_sizes = []
        if num_devices > 0:
            quotient, remainder = divmod(self.output_size, num_devices)
            start_index = 0
            for i in range(1, num_devices + 1):
                end_index = i * quotient + min(i, remainder)
                self._weight_sizes.append(end_index - start_index)
                start_index = end_index
            assert len(self._weight_sizes) == num_devices
            assert sum(self._weight_sizes) == self.output_size
            assert end_index == self.output_size

        logging.debug("- %s name=%s inputs=[%s] sizes=[%s], devices=[%s]",
            self.__class__.__name__,
            self.name,
            ', '.join([x.name for x in self.input_layers]),
            ', '.join([str(x) for x in self._weight_sizes]),
            ', '.join([str(x) for x in self._devices]))

        self._network = network
        self._profile = profile
        self.param_init_values = OrderedDict()

    def set_params(self, params):
        """Sets the dictionary that can be used to access Theano shared
        variables.

        :type params: dict
        :param params: a dictionary of Theano shared variables indexed by
                       parameter name
        """

        self._params = params

    @abstractmethod
    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        assert False

    def _param_path(self, param_name):
        """Returns the HDF5 path used to address a parameter.

        :type param_name: str
        :param param_name: name of a parameter within this layer

        :rtype: str
        :returns: full path of the parameter in a HDF5 file.
        """

        return 'layers/' + self.name + '/' + param_name

    def _get_param(self, param_name):
        """Returns a Theano tensor variable by parameter name.

        :type param_name: str
        :param param_name: name of a parameter within the layer

        :rtype: TensorVariable
        :returns: the corresponding tensor variable
        """

        return self._params[self._param_path(param_name)]

    def _init_weight(self, param_name, shape, scale=None, count=1):
        """Generates a weight matrix from “standard normal” distribution.

        If ``shape`` contains two dimensions that match, generates an orthogonal
        matrix. In that case scale is ignored. Orthogonal weights are useful for
        two reasons:

        1. Multiplying by an orthogonal weight preserves the norm of the
           input vector, which should help avoid exploding and vanishing
           gradients.
        2. The row and column vectors are orthonormal to one another, which
           should help avoid two vectors learning to produce the same features.

        :type shape: list or tuple of ints
        :param shape: sizes of the weight dimensions; normally the first one is
                      the dimensionality of the input data and the second one is
                      the dimensionality of the output data

        :type scale: float
        :param scale: if other than ``None``, the matrix will be scaled by this
                      factor, unless an orthogonal matrix is created
        """

        weight_path = self._param_path(param_name)
        if (len(shape) == 2) and (shape[0] == shape[1]):
            weight_matrix = numpy.concatenate(
                [orthogonal_weight(shape[0]) for _ in range(count)],
                axis=1)
        else:
            weight_matrix = numpy.concatenate(
                [random_weight(shape, scale) for _ in range(count)],
                axis=1)
        self.param_init_values[weight_path] = weight_matrix

    def _init_bias(self, param_name, shape, value=None):
        """Initializes a bias vector with given value.

        If ``value`` is not given, initializes the vector with zero value. If
        ``value`` is a list, creates a concatenation of as many vectors as there
        are elements in the list.

        :type param_name: str
        :param param_name: name for the parameter within the layer

        :type shape: int or tuple of ints
        :param shape: size of the vector, or a tuple of the sizes of each
                      dimension (in case ``value`` is a list, each part will
                      have this size)

        :type value: float, numpy.ndarray or list
        :param value: the value or array to initialize the elements to, or a
                      list of values or arrays to create a concatenation of
                      vectors
        """

        values = value if isinstance(value, (list, tuple)) else [value]
        parts = []
        for part_value in values:
            if part_value is None:
                part = numpy.zeros(shape).astype(theano.config.floatX)
            elif isinstance(value, numpy.ndarray):
                part = value.astype(theano.config.floatX)
            else:
                part = numpy.empty(shape).astype(theano.config.floatX)
                part.fill(part_value)
            parts.append(part)
        self.param_init_values[self._param_path(param_name)] = \
            numpy.concatenate(parts)

    def _tensor_preact(self, input_matrix, param_name):
        """Helper function that creates a pre-activation of ``input_matrix`` by
        multiplying it by a weight matrix and adding a bias.

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
        :param param_name: name of a parameter group that contains a weight
                           matrix and a bias vector

        :rtype: TensorVariable
        :returns: a matrix tha has the same number of dimensions as
                  ``input_matrix``, but the data vectors (the last dimension of
                  this matrix) are the preactivations
        """

        weight = self._params[self._param_path(param_name) + '/W']
        bias = self._params[self._param_path(param_name) + '/b']
        return tensor.dot(input_matrix, weight) + bias
