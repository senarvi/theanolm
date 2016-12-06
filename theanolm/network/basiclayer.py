#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
from collections import OrderedDict
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.weightfunctions import weight_matrix

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

        logging.debug("- %s name=%s inputs=[%s] size=%d, devices=[%s]",
            self.__class__.__name__,
            self.name,
            ', '.join([x.name for x in self.input_layers]),
            self.output_size,
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

    def _param_path(self, param_name, device=None):
        """Returns the HDF5 path used to address a parameter.

        :type param_name: str
        :param param_name: name of a parameter within this layer

        :type device: str
        :param device: ``None`` for parameters that reside on the default device
                       only; otherwise returns the path used to address the part
                       of the parameter that resides on the given device

        :rtype: str
        :returns: full path of the parameter in a HDF5 file.
        """

        result = 'layers/' + self.name + '/' + param_name
        if not device is None:
            result += '/' + device
        return result

    def _get_param(self, param_name, device=None):
        """Returns a Theano tensor variable by parameter name.

        :type param_name: str
        :param param_name: name of a parameter within the layer

        :type device: str
        :param device: ``None`` for parameters that reside on the default device
                       only; otherwise returns the part of the parameter that
                       resides on the given device

        :rtype: TensorVariable
        :returns: the corresponding tensor variable
        """

        return self._params[self._param_path(param_name, device)]

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

        If ``count`` is specified, creates a concatenation of several similar
        matrices (same shape but different content).

        :type shape: list or tuple of ints
        :param shape: sizes of the weight dimensions; normally the first one is
                      the dimensionality of the input data and the second one is
                      the dimensionality of the output data

        :type scale: float
        :param scale: if other than ``None``, the matrix will be scaled by this
                      factor, unless an orthogonal matrix is created

        :type count: int
        :param count: concatenate this many weight matrices with the same shape
        """

        path = self._param_path(param_name)
        self.param_init_values[path] = weight_matrix(shape, scale, count)

    def _init_split_weight(self, param_name, shape, scale=None, count=1):
        """Generates one weight matrix per device from “standard normal”
        distribution. The last dimension in ``shape`` is divided by the number
        of devices.

        If ``shape`` contains two dimensions that match, generates an orthogonal
        matrix. In that case scale is ignored. Orthogonal weights are useful for
        two reasons:

        1. Multiplying by an orthogonal weight preserves the norm of the
           input vector, which should help avoid exploding and vanishing
           gradients.
        2. The row and column vectors are orthonormal to one another, which
           should help avoid two vectors learning to produce the same features.

        If ``count`` is specified, the created matrices will be concatenations
        of several similar matrices (the last dimension of each submatrix is
        divided by the number of devices).

        :type shape: list or tuple of ints
        :param shape: sizes of the weight dimensions; normally the first one is
                      the dimensionality of the input data and the second one is
                      the dimensionality of the output data

        :type scale: float
        :param scale: if other than ``None``, the matrix will be scaled by this
                      factor, unless an orthogonal matrix is created

        :type count: int
        :param count: concatenate this many weight matrices with the same shape
        """

        if (len(self._devices) == 1) and (self._devices[0] == None):
            # This layer has not been assigned to a specific device.
            return self._init_weight(param_name, shape, scale, count)

        sizes = self._size_per_device(shape[-1])
        for device, size in self._devices, sizes:
            assert not device is None
            path = self._param_path(param_name) + '/' + device
            shape[-1] = size
            self.param_init_values[path] = weight_matrix(shape, scale, count)

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

    def _size_per_device(self, total_size):
        """Returns ``total_size`` divided for each device.

        :type total_size: int
        :param total_size: total size of a parameter

        :rtype: list of ints
        :returns: ``total_size`` divided into as many parts as there are devices
                  assigned to this layer
        """

        num_devices = len(self._devices)
        if num_devices < 1:
            raise RuntimeError("No devices assigned to this layer.")

        result = []
        quotient, remainder = divmod(total_size, num_devices)
        start_index = 0
        for i in range(1, num_devices + 1):
            end_index = i * quotient + min(i, remainder)
            result.append(end_index - start_index)
            start_index = end_index

        assert len(result) == num_devices
        assert sum(result) == total_size
        assert end_index == total_size

        return result

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
