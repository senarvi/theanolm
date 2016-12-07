#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm import Parameters
from theanolm.network.weightfunctions import random_matrix, matrix_from_value

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
        self.params = Parameters()
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

    @abstractmethod
    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer.
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

        return self.params[self._param_path(param_name, device)]

    def _init_weight(self, param_name, shape, scale=None, count=1,
                     split_to_devices=False):
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
        submatrices (same shape but different content).

        If ``split_to_devices`` is set to ``True``, splits the weight to equal
        parts on the last dimension, and creates one parameter for each device.
        If also ``count`` is specified, each device will have an equal part of
        every submatrix.

        :type shape: list or tuple of ints
        :param shape: sizes of the weight dimensions; normally the first one is
                      the dimensionality of the input data and the second one is
                      the dimensionality of the output data

        :type scale: float
        :param scale: if other than ``None``, the matrix will be scaled by this
                      factor, unless an orthogonal matrix is created

        :type count: int
        :param count: concatenate this many weight matrices with the same shape

        :type split_to_devices: bool
        :param split_to_devices: if set to ``True``, creates on every device a
                                 parameter that contains one part of the weight
        """

        path = self._param_path(param_name)
        weight = random_matrix(shape, scale, count)
        if not split_to_devices:
            self.params.add(path, random_matrix(shape, scale, count))
        elif (len(self._devices) == 1) and (self._devices[0] == None):
            # This layer has not been assigned to a specific device.
            self.params.add(path, random_matrix(shape, scale, count))
        else:
            self._split_to_devices(path, weight, shape[-1])

    def _init_bias(self, param_name, shape, value=None, split_to_devices=False):
        """Initializes a bias vector with given value.

        If ``value`` is not given, initializes the vector with zero value. If
        ``value`` is a list, creates a concatenation of as many vectors as there
        are elements in the list.

        If ``split_to_devices`` is set to ``True``, splits the array to equal
        parts on the last dimension, and creates one parameter for each device.
        If ``value`` is a list, each device will have an equal part of every
        submatrix.

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

        :type split_to_devices: bool
        :param split_to_devices: if set to ``True``, creates on every device a
                                 parameter that contains one part of the array
        """

        path = self._param_path(param_name)
        bias = matrix_from_value(shape, value)
        if not split_to_devices:
            self.params.add(path, matrix_from_value(shape, value))
        elif (len(self._devices) == 1) and (self._devices[0] == None):
            # This layer has not been assigned to a specific device.
            self.params.add(path, matrix_from_value(shape, value))
        else:
            self._split_to_devices(path, bias, shape[-1])

    def _split_to_devices(self, path, value, part_size):
        """Splits a matrix to equal parts on the last dimension, and creates a
        parameter on each device.

        If the matrix consists of submatrices, each device will have an equal
        part of every submatrix, whose size is specified by ``part_size``.

        :type path: str
        :param path: base path for the parameters that will be prefixed by the
                     device string

        :type value: numpy.ndarray
        :param value: a matrix that will be split to give the initial value of
                      the parameters

        :type part_size: int
        :param part_size: size of the last dimension of ``value``, or if
                          ``value`` consists of multiple submatrices, size of
                          one submatrix
        """

        part_count = value.shape[-1] // part_size
        if part_count * part_size != value.shape[-1]:
            raise ValueError("Last dimension is not a multiple of part size.")

        split_sizes = self._size_per_device(part_size)
        split_start = 0
        for device, split_size in zip(self._devices, split_sizes):
            assert not device is None
            split_end = split_start + split_size
            ranges = []
            for part_index in range(part_count):
                part_start = part_index * part_size
                ranges.extend(range(part_start + split_start,
                                    part_start + split_end))
            split_start = split_end
            self.params.add(path + '/' + device, value[..., ranges], device)

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
        if total_size < num_devices:
            raise ValueError("Cannot split matrix of size {} to {} devices."
                             .format(total_size, num_devices))

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

        weight = self.params[self._param_path(param_name) + '/W']
        bias = self.params[self._param_path(param_name) + '/b']
        return tensor.dot(input_matrix, weight) + bias
