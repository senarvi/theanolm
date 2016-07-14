#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.basiclayer import BasicLayer

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
        # independently for each location, over the output classes. This
        # produces probabilities for the whole vocabulary.
        num_time_steps = preact.shape[0]
        num_sequences = preact.shape[1]
        output_size = preact.shape[2]
        preact = preact.reshape([num_time_steps * num_sequences,
                                 output_size])
        self.output_probs = tensor.nnet.softmax(preact)
        self.output_probs = self.output_probs.reshape([num_time_steps,
                                                       num_sequences,
                                                       output_size])
        if self.network.predict_next_distribution:
            return

        # We should predict probabilities of the target outputs, i.e. the words
        # at the next time step.
        output_probs = self.output_probs[:-1].flatten()
        class_ids = self.network.class_input[1:].flatten()

        # An index to a flattened input matrix times the vocabulary size can be
        # used to index the same location in the output matrix. The class ID is
        # added to index the probability of that word.
        minibatch_size = class_ids.shape[0]
        num_classes = self.network.vocabulary.num_classes()
        target_indices = tensor.arange(minibatch_size) * num_classes + class_ids
        self.target_probs = output_probs[target_indices]
        self.target_probs = self.target_probs.reshape([(num_time_steps - 1),
                                                       num_sequences])
