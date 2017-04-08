#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import copy
import theano.tensor as tensor
from theanolm.network.grulayer import GRULayer
from theanolm.network.lstmlayer import LSTMLayer

class BidirectionalLayer(object):
    """Wrapper for Combining Forward and Backward Recurrent Layers

    M. Schuster, K. K. Paliwal
    Bidirectional Recurrent Neural Networks
    IEEE Transactions on Signal Processing, 45(11), 2673–2681

    Combines two recurrent layers, one which has dependency forward in time, and
    one with dependency backward in time. The input of the backward layer is
    shifted two time steps to make sure the target word is not predicted using
    itself (which is also the next input word). Note that the probability of a
    word depends on the future words as well, instead of just the past words.
    Thus the sequence probabilities are not a true probability distribution, and
    text cannot be generated.
    """

    def __init__(self, layer_options, *args, **kwargs):
        layer_type = layer_options['type']
        self.name = layer_options['name']
        if 'size' in layer_options:
            self.output_size = int(layer_options['size'])
        else:
            input_layers = layer_options['input_layers']
            self.output_size = sum([x.output_size for x in input_layers])
        backward_size = self.output_size // 2
        forward_size = self.output_size - backward_size

        forward_options = layer_options
        backward_options = copy(layer_options)
        forward_options['name'] = self.name + '/forward'
        forward_options['size'] = forward_size
        backward_options['name'] = self.name + '/backward'
        backward_options['size'] = backward_size
        backward_options['reverse_time'] = True
        if layer_type == 'blstm':
           self._forward_layer = LSTMLayer(forward_options, *args, **kwargs)
           self._backward_layer = LSTMLayer(backward_options, *args, **kwargs)
        elif layer_type == 'bgru':
           self._forward_layer = GRULayer(forward_options, *args, **kwargs)
           self._backward_layer = GRULayer(backward_options, *args, **kwargs)
        else:
            raise ValueError("Invalid layer type requested: " + layer_type)

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets self.output to a symbolic matrix that describes the output of this
        layer.
        """

        self._forward_layer.create_structure()
        self._backward_layer.create_structure()
        self.output = tensor.concatenate([self._forward_layer.output,
                                          self._backward_layer.output],
                                         axis=2)

    def get_state(self, state):
        """Pulls parameter values from Theano shared variables.

        If there already is a parameter in the state, it will be replaced, so it
        has to have the same number of elements.

        :type state: h5py.File
        :param state: HDF5 file for storing the neural network parameters
        """

        self._forward_layer._params.get_state(state)
        self._backward_layer._params.get_state(state)

    def set_state(self, state):
        """Sets the values of Theano shared variables.

        :type state: h5py.File
        :param state: HDF5 file that contains the neural network parameters
        """

        self._forward_layer._params.set_state(state)
        self._backward_layer._params.set_state(state)

    def num_params(self):
        """Returns the number of parameters in this layer.

        This method is used just for reporting the number of parameters in the
        model. Normally there is just one set of parameters.

        :rtype: int
        :returns: the number of parameters used by the layer
        """

        return self._forward_layer._params.total_size + \
               self._backward_layer._params.total_size

    def get_variables(self):
        """Returns a dictionary of the shared variables.

        This function is used by the optimizers to create optimization
        parameters that are specific to network parameters, and compute
        gradients with regard to the parameters.

        :rtype: dict
        :returns: mapping from parameter path to Theano shared variables
        """

        result = dict()
        result.update(self._forward_layer.get_variables())
        result.update(self._backward_layer.get_variables())
        return result
