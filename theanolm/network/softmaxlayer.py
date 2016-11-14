#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.ncelayer import NCELayer
from theanolm.debugfunctions import *

class SoftmaxLayer(NCELayer):
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
        output_probs = tensor.nnet.softmax(preact)
        output_probs = output_probs.reshape([num_time_steps,
                                             num_sequences,
                                             output_size])
        self.output_probs = output_probs

        if self.network.target_class_ids is None:
            self.target_probs = None
            self.unnormalized_logprobs = None
            self.sample = None
            self.sample_logprobs = None
            self.shared_sample = None
            self.shared_sample_logprobs = None
            return

        num_time_steps = output_probs.shape[0]
        num_sequences = output_probs.shape[1]
        target_class_ids = self.network.target_class_ids
        target_class_ids = assert_tensor_eq(
            target_class_ids,
            'target_class_ids.shape[0]',
            'num_time_steps',
            target_class_ids.shape[0],
            num_time_steps)
        target_class_ids = assert_tensor_eq(
            target_class_ids,
            'target_class_ids.shape[1]',
            'num_sequences',
            target_class_ids.shape[1],
            num_sequences)
        num_classes = self.network.vocabulary.num_classes()
        assert num_classes == self.output_size

        # Compute unnormalized output and noise samples for NCE.
        self._compute_unnormalized_logprobs(layer_input)
        self._compute_sample_logprobs(layer_input)

        # An index to a flattened input matrix times the vocabulary size can be
        # used to index the same location in the output matrix. The class ID is
        # added to index the probability of that word.
        output_probs = output_probs.flatten()
        target_class_ids = target_class_ids.flatten()
        minibatch_size = target_class_ids.shape[0]
        output_probs = assert_tensor_eq(
            output_probs,
            'output_probs.shape[0]',
            'minibatch_size * num_classes',
            output_probs.shape[0],
            minibatch_size * num_classes)
        target_indices = tensor.arange(minibatch_size) * num_classes
        target_indices += target_class_ids
        self.target_probs = output_probs[target_indices]
        self.target_probs = self.target_probs.reshape([num_time_steps,
                                                       num_sequences])
