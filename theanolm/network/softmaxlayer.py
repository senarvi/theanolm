#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.samplingoutputlayer import SamplingOutputLayer
from theanolm.debugfunctions import *

class SoftmaxLayer(SamplingOutputLayer):
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
        self._init_weight('input/W', (input_size, output_size), scale=0.01)
        if self._network.class_prior_probs is None:
            self._init_bias('input/b', output_size)
        else:
            initial_bias = numpy.log(self._network.class_prior_probs + 1e-10)
            self._init_bias('input/b', output_size, initial_bias)

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        word projection. When generating text, there's just one sequence and one
        time step in the input.

        Sets self.output to a symbolic matrix that describes the output of this
        layer.
        """

        layer_input = tensor.concatenate([x.output for x in self.input_layers],
                                         axis=2)
        preact = self._tensor_preact(layer_input, 'input')

        # Combine the first two dimensions so that softmax is taken
        # independently for each location, over the output classes. This
        # produces probabilities for the whole vocabulary.
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        preact = preact.reshape([num_time_steps * num_sequences,
                                 self.output_size])
        output_probs = tensor.nnet.softmax(preact)
        self.output_probs = output_probs.reshape([num_time_steps,
                                                  num_sequences,
                                                  self.output_size])

        # The following variables can only be used when
        # self._network.target_class_ids is given to the function.
        element_ids = tensor.arange(num_time_steps * num_sequences)
        target_class_ids = self._network.target_class_ids.flatten()
        target_probs = output_probs[(element_ids, target_class_ids)]
        self.target_probs = target_probs.reshape([num_time_steps,
                                                  num_sequences])

        # Compute unnormalized output and noise samples for NCE.
        self.unnormalized_logprobs = \
            self._get_unnormalized_logprobs(layer_input)
        self.sample, self.sample_logprobs = \
            self._get_sample_tensors(layer_input)
        self.seqshared_sample, self.seqshared_sample_logprobs = \
            self._get_seqshared_sample_tensors(layer_input)
        self.shared_sample, self.shared_sample_logprobs = \
            self._get_shared_sample_tensors(layer_input)
