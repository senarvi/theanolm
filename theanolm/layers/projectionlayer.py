#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.layers.basiclayer import BasicLayer

class ProjectionLayer(BasicLayer):
    """Projection Layer for Neural Network Language Model
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(*args, **kwargs)

        # Initialize the parameters.
        input_size = self.input_layers[0].output_size
        output_size = self.output_size
        self._init_orthogonal_weight('W', input_size, output_size, scale=0.01)

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 2-dimensional: the first dimension is the time step
        (index of word in a sequence) and the second dimension are the
        sequences. When generating text, there's just one sequence and one time
        step in the input.

        Sets self.output to a symbolic matrix that describes the output of this
        layer. Assumes that the shared variables have been passed using
        ``set_params()``.
        """

        input_matrix = self.input_layers[0].output
        num_time_steps = input_matrix.shape[0]
        num_sequences = input_matrix.shape[1]

        # Indexing the word_projection matrix with a word ID returns the
        # self.output_size dimensional projection. Note that indexing the
        # matrix with a vector of all the word IDs gives a concatenation of
        # those projections.
        projections = self._get_param('W')[input_matrix.flatten()]
        projections = projections.reshape([num_time_steps,
                                           num_sequences,
                                           self.output_size],
                                          ndim=3)
        self.output = projections
