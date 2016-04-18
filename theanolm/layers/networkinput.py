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

    def __init__(self, input_options, network):
        """Creates a neural network input with a given vocabulary size, which
        specifies the input size of the first layer.

        :type input_options: dict
        :param input_options: dictionary of input options

        :type network: Network
        :param network: the network object which uses this input
        """

        self.input_type = input_options['type']
        if self.input_type == 'word':
            output_size = network.vocabulary.num_words()
        elif self.input_type == 'class':
            output_size = network.vocabulary.num_classes()
        else:
            raise ValueError(
                "Invalid network input type: {}".format(self.input_type))
        input_options['size'] = output_size
        input_options['input_layers'] = []
        super().__init__(input_options, network)

    def create_structure(self):
        """Creates the symbolic matrix that describes the network input.

        The tensor variable will be set to a matrix of word IDs, with
        [ number of time steps * number of sequences ] elements. When generating
        text, the matrix will contain only one element.
        """

        if self.input_type == 'word':
            self.output = self.network.word_input
        elif self.input_type == 'class':
            self.output = self.network.class_input
        else:
            assert False
