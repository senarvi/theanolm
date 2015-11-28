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

    def __init__(self, output_size):
        """Initializes neural network input.

        :type output_size: int
        :param output_size: number of output connections
        """

        super().__init__('__input__', [], output_size)

    def create_structure(self):
        """Creates the symbolic matrix that describes the network input.

        The tensor variable will be set to a matrix of word IDs, with
        [ number of time steps * number of sequences ] elements. When generating
        text, the matrix will contain only one element.
        """

        self.output = tensor.matrix('network_input', dtype='int64')
        self.output.tag.test_value = test_value(
            size=(100, 16),
            max_value=self.output_size)
