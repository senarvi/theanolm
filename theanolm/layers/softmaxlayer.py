#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.layers.basiclayer import BasicLayer

class SoftmaxLayer(BasicLayer):
    """ Output Layer for Neural Network Language Model

    The output layer is a simple softmax layer that outputs the word
    probabilities.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(*args, **kwargs)

        # Create the parameters. Weight matrix and bias for each input.
        output_size = self.output_size
        for input_index, input_layer in enumerate(self.input_layers):
            input_size = input_layer.output_size
            param_name = 'input' + str(input_index) + '/W'
            self._init_random_weight(param_name, input_size, output_size, scale=0.01)
            param_name = 'input' + str(input_index) + '/b'
            self._init_bias(param_name, output_size)

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        word projection. When generating text, there's just one sequence and one
        time step in the input.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        preacts = []
        for input_index, input_layer in enumerate(self.input_layers):
            param_name = 'input' + str(input_index)
            preacts.append(self._tensor_preact(input_layer.output, param_name))
        preact = sum(preacts)

        # Combine the first two dimensions so that softmax is taken
        # independently for each location, over the output classes.
        num_time_steps = preact.shape[0]
        num_sequences = preact.shape[1]
        num_classes = preact.shape[2]
        preact = preact.reshape([num_time_steps * num_sequences,
                                 num_classes])
        self.output = tensor.nnet.softmax(preact)
        self.output = self.output.reshape([num_time_steps,
                                           num_sequences,
                                           num_classes])
