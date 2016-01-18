#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import test_value
from theanolm.layers.basiclayer import BasicLayer

class NetworkInput(BasicLayer):
    """Input for Neural Network Language Model

    A dummy layer that provides the input for the first layer.
    """

    def __init__(self, output_size, network):
        """Creates a neural network input with a given vocabulary size, which
        specifies the input size of the first layer.

        :type output_size: int
        :param output_size: number of output connections

        :type network: Network
        :param network: the network object creating this layer
        """

        layer_options = { 'name': '__input__',
                          'input_layers': [],
                          'output_size': output_size }
        super().__init__(layer_options, network)

    def create_structure(self):
        """Creates the symbolic matrix that describes the network input.

        The tensor variable will be set to a matrix of word IDs, with
        [ number of time steps * number of sequences ] elements. When generating
        text, the matrix will contain only one element.
        """

        self.output = tensor.matrix('network.input', dtype='int64')
        self.output.tag.test_value = test_value(
            size=(100, 16),
            max_value=self.output_size)
