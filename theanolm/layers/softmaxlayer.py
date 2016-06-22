#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.layers.basiclayer import BasicLayer

class SoftmaxLayer(BasicLayer):
    """Softmax Output Layer

    The output layer is a simple softmax layer that outputs the word
    probabilities.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(*args, **kwargs)

        # Create the parameters. Weight matrix and bias for each input.
        input_size = sum(x.output_size for x in self.input_layers)
        output_size = self.output_size
        self._init_random_weight('input/W',
                                 (input_size, output_size),
                                 scale=0.01)
        self._init_bias('input/b', output_size)

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

        layer_input = tensor.concatenate([x.output for x in self.input_layers],
                                         axis=2)
        preact = self._tensor_preact(layer_input, 'input')

        # Combine the first two dimensions so that softmax is taken
        # independently for each location, over the output classes.
        num_time_steps = preact.shape[0]
        num_sequences = preact.shape[1]
        output_size = preact.shape[2]
        preact = preact.reshape([num_time_steps * num_sequences,
                                 output_size])
        output_probs = tensor.nnet.softmax(preact)
        output_probs = output_probs.reshape([num_time_steps,
                                           num_sequences,
                                           output_size])

        # The input at the next time step is what the output (predicted word)
        # should be.
        class_ids = self.network.class_input[1:].flatten()
        output_probs = output_probs[:-1].flatten()

        # An index to a flattened input matrix times the vocabulary size can be
        # used to index the same location in the output matrix. The class ID is
        # added to index the probability of that word.
        target_indices = \
            tensor.arange(class_ids.shape[0]) * self.network.vocabulary.num_classes() \
            + class_ids
        self.output = output_probs[target_indices]

        # Reshape to a matrix. Now we have one less time step.
        self.output = self.output.reshape([(num_time_steps - 1), num_sequences])
