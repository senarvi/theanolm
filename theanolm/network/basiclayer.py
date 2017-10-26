#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the BasicLayer class, a base class for all layers.
"""

from abc import abstractmethod, ABCMeta
import logging

import theano.tensor as tensor

from theanolm.backend import Parameters, conv1d
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
        self._input_layers = layer_options['input_layers']
        self._params = Parameters()
        self._devices = layer_options['devices']

        if 'size' in layer_options:
            self.output_size = int(layer_options['size'])
        else:
            self.output_size = \
                sum([x.output_size for x in self._input_layers])

        if 'activation' in layer_options:
            activation = layer_options['activation']
        else:
            activation = 'tanh'
        if activation == 'tanh':
            self._activation = tensor.tanh
        elif activation == 'relu':
            self._activation = tensor.nnet.relu
        elif activation == 'leakyrelu':
            self._activation = lambda x: tensor.nnet.relu(x, 0.3)
        else:
            raise InputError("Unsupported activation function: {}"
                             .format(activation))

        if 'reverse_time' in layer_options:
            self._reverse_time = bool(layer_options['reverse_time'])
        else:
            self._reverse_time = False

        logging.debug("- %s name=%s inputs=[%s] size=%d activation=%s%s "
                      "devices=[%s]",
                      self.__class__.__name__,
                      self.name,
                      ', '.join([x.name for x in self._input_layers]),
                      self.output_size,
                      activation,
                      ' reverse,' if self._reverse_time else '',
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

    def get_state(self, state):
        """Pulls parameter values from Theano shared variables.

        If there already is a parameter in the state, it will be replaced, so it
        has to have the same number of elements.

        :type state: h5py.File
        :param state: HDF5 file for storing the neural network parameters
        """

        self._params.get_state(state)

    def set_state(self, state):
        """Sets the values of Theano shared variables.

        :type state: h5py.File
        :param state: HDF5 file that contains the neural network parameters
        """

        self._params.set_state(state)

    def num_params(self):
        """Returns the number of parameters in this layer.

        This method is used just for reporting the number of parameters in the
        model. Normally there is just one set of parameters.

        :rtype: int
        :returns: the number of parameters used by the layer
        """

        return self._params.total_size

    def get_variables(self):
        """Returns a dictionary of the shared variables.

        This function is used by the optimizers to create optimization
        parameters that are specific to network parameters, and compute
        gradients with regard to the parameters. Normally there is just one set
        of parameters.

        :rtype: dict
        :returns: mapping from parameter path to Theano shared variables
        """

        return self._params.get_variables()

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
        if device is not None:
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

        :rtype: Variable
        :returns: the corresponding tensor variable
        """

        return self._params[self._param_path(param_name, device)]

    def _init_weight(self, param_name, shape, scale=None, count=1,
                     split_to_devices=True):
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

        If ``split_to_devices`` is ``True``, splits the weight to equal parts on
        the last dimension, and creates one parameter for each device.

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
        :param split_to_devices: if ``True``, creates on every device a
                                 parameter that contains one part of the weight
        """

        path = self._param_path(param_name)
        weight = random_matrix(shape, scale, count)
        if not split_to_devices:
            self._params.add(path, weight)
        elif (len(self._devices) == 1) and (self._devices[0] is None):
            # This layer has not been assigned to a specific device.
            self._params.add(path, weight)
        else:
            split_ranges = self._split_per_device(weight.shape[-1])
            for device, split_range in zip(self._devices, split_ranges):
                assert device is not None
                self._params.add('{}/{}'.format(path, device),
                                 weight[..., split_range], device)

    def _init_bias(self, param_name, shape, value=None, split_to_devices=True):
        """Initializes a bias vector with given value.

        If ``value`` is not given, initializes the vector with zero value. If
        ``value`` is a list, creates a concatenation of as many vectors as there
        are elements in the list.

        If ``split_to_devices`` is ``True``, splits the array to equal parts on
        the last dimension, and creates one parameter for each device.

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
        :param split_to_devices: if ``True``, creates on every device a
                                 parameter that contains one part of the array
        """

        path = self._param_path(param_name)
        bias = matrix_from_value(shape, value)
        if not split_to_devices:
            self._params.add(path, matrix_from_value(shape, value))
        elif (len(self._devices) == 1) and (self._devices[0] is None):
            # This layer has not been assigned to a specific device.
            self._params.add(path, matrix_from_value(shape, value))
        else:
            split_ranges = self._split_per_device(bias.shape[-1])
            for device, split_range in zip(self._devices, split_ranges):
                assert device is not None
                self._params.add('{}/{}'.format(path, device),
                                 bias[..., split_range], device)

    def _split_per_device(self, total_size):
        """Returns a list of range objects that divide ``total_size`` equally to
        as many parts as there are devices.

        :type total_size: int
        :param total_size: total size of a parameter

        :rtype: list of ranges
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
            result.append(range(start_index, end_index))
            start_index = end_index

        assert len(result) == num_devices
        assert sum(len(x) for x in result) == total_size
        assert end_index == total_size

        return result

    def _tensor_preact(self, input_matrix, param_name, use_bias=True):
        """Helper function that creates a pre-activation of ``input_matrix`` by
        multiplying it by a weight matrix and adding a bias.

        ``input_matrix`` and the result normally have the shape of a mini-batch:
        the first dimension is the time step and the second dimension is the
        sequence. The last dimension is always the features. The preactivations
        for each mini-batch element will be computed by multiplying the features
        by the weight matrix and adding the bias. Thus the dimensionality of the
        input features should equal to the first dimension of the weight matrix,
        and the second dimension of the weight matrix defines the dimensionality
        of the output features.

        :type input_matrix: symbolic tensor
        :param input_matrix: one or more sequences of features in the shape
                             (time steps, sequences, features)

        :type param_name: str
        :param param_name: name of a parameter group that contains a weight
                           matrix and a bias vector

        :type use_bias: bool
        :param use_bias: if set to ``False``, does not add a bias

        :rtype: symbolic tensor
        :returns: a tensor that has the same number of dimensions as
                  ``input_matrix``, but the features (the last dimension) are
                  the preactivations
        """

        results = []
        for device in self._devices:
            weight = self._get_param(param_name + '/W', device)
            result = tensor.dot(input_matrix, weight)
            if use_bias:
                bias = self._get_param(param_name + '/b', device)
                result += bias
            results.append(result)

        if len(results) > 1:
            return tensor.concatenate(results, axis=2)
        elif len(results) == 1:
            return results[0]
        else:
            assert False

    def _tensor_conv1d(self, input_matrix, param_name):
        """Convolves ``input_matrix`` using filters and adds a bias.

        ``input_matrix`` and the result normally have the shape of a mini-batch:
        the first dimension is the time step and the second dimension is the
        sequence. The last dimension is always the features.

        The filter tensor is a stack of filters, each producing one output
        feature. One output feature in one mini-batch location will be computed
        by multiplying the input features in a local region by the corresponding
        filter elements, taking the sum over the input features in the region,
        and adding the bias. The first dimension of the filter defines the size
        of the region, the second dimension should equal to the dimensionality
        of the input features, and the third dimension defines the
        dimensionality of the output features.

        :type input_matrix: symbolic 3D tensor
        :param input_matrix: one or more sequences of features in the shape
                             (time steps, sequences, features)

        :type param_name: str
        :param param_name: name of a parameter group that contains a filter
                           matrix and a bias vector

        :rtype: symbolic 3D tensor
        :returns: the input convolved with the filters in the shape (time steps,
                  sequences, features)
        """

        # Permutate input dimensions from (time steps, sequences, features) to
        # (samples, elements, features).
        input_matrix = input_matrix.dimshuffle(1, 0, 2)

        results = []
        for device in self._devices:
            filters = self._get_param(param_name + '/W', device)
            result = conv1d(input_matrix,
                            filters,
                            padding='valid')
            results.append(result)

        if len(results) > 1:
            result = tensor.concatenate(results, axis=2)
        elif len(results) == 1:
            result = results[0]
        else:
            assert False

        # Permutate input dimensions from (samples, elements, features) to
        # (time steps, sequences, features).
        result = result.dimshuffle(1, 0, 2)

        result += self._get_param(param_name + '/b')
        return result
