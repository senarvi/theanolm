#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.matrixfunctions import orthogonal_weight

class InputError(Exception):
    """Exception raised for errors in the input.
    """
    pass

class ProjectionLayer(object):
    """Projection Layer for Neural Network Language Model
    """

    def __init__(self, in_size, out_size):
        """Initializes the parameters for the first layer of a neural network
        language model, which creates the word embeddings.

        :type in_size: int
        :param in_size: dimensionality of the input vectors, i.e. vocabulary
                        size

        :type out_size: int
        :param out_size: dimensionality of the word projections
        """

        self.word_projection_dim = out_size

        # Initialize the parameters.
        self.param_init_values = OrderedDict()

        self.param_init_values['proj.W'] = \
                orthogonal_weight(in_size, out_size, scale=0.01)

    def create_structure(self, model_params, layer_input):
        """Creates projection layer structure.

        The input is always 2-dimensional: the first dimension is the time step
        (index of word in a sequence) and the second dimension are the
        sequences. When generating text, there's just one sequence and one time
        step in the input.

        :type model_params: dict
        :param model_params: shared Theano variables

        :type layer_input: theano.tensor.var.TensorVariable
        :param layer_input: symbolic 2-dimensional matrix that describes
                            the input

        :rtype: theano.tensor.var.TensorVariable
        :returns: symbolic 3-dimensional matrix - the first dimension is
                  the time step, the second dimension are the sequences,
                  and the third dimension is the word projection
        """

        print("ProjectionLayer.create_structure: layer_input.ndim =", layer_input.ndim)
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]

        # Indexing the word_projection matrix with a word ID returns the
        # word_projection_dim dimensional projection. Note that indexing the
        # matrix with a vector of all the word IDs gives a concatenation of
        # those projections.
        projections = model_params['proj.W'][layer_input.flatten()]
        projections = projections.reshape([num_time_steps,
                                           num_sequences,
                                           self.word_projection_dim],
                                          ndim=3)
        self.output = projections
