#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import orthogonal_weight

class OutputLayer(object):
    """ Output Layer for Neural Network Language Model

    The output layer is a simple softmax layer that outputs the word
    probabilities.
    """

    def __init__(self, in_size, out_size):
        """Initializes the parameters for a feed-forward layer of a neural
        network.

        :type in_size: int
        :param in_size: number of input connections
        
        :type out_size: int
        :param out_size: size of the output
        """

        # Create the parameters.
        self.param_init_values = OrderedDict()

        self.param_init_values['output.W'] = \
                orthogonal_weight(in_size, out_size, scale=0.01)

        self.param_init_values['output.b'] = \
                numpy.zeros((out_size,)).astype(theano.config.floatX)

    def create_structure(self, model_params, layer_input):
        """ Creates output layer structure.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        word projection. When generating text, there's just one sequence and one
        time step in the input.

        Sets self.output to a symbolic 3-dimensional matrix that describes the
        output of this layer, i.e. the probability of every vocabulary word for
        each input.

        :type model_params: dict
        :param model_params: shared Theano variables

        :type layer_input: theano.tensor.var.TensorVariable
        :param layer_input: symbolic matrix that describes the output of the
        previous layer
        """

        preact = tensor.dot(layer_input, model_params['output.W']) \
                + model_params['output.b']

        num_time_steps = preact.shape[0]
        num_sequences = preact.shape[1]
        num_classes = preact.shape[2]
        # Combine the first two dimensions so that softmax is taken
        # independently for each location over the output classes.
        preact = preact.reshape([num_time_steps * num_sequences,
                                 num_classes])

        self.output = tensor.nnet.softmax(preact)
