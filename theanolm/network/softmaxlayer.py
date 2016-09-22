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

        num_classes = self.network.vocabulary.num_classes()
        assert num_classes == self.output_size

        layer_input = tensor.concatenate([x.output for x in self.input_layers],
                                         axis=2)

        if self.network.mode.is_distribution():
            self.output_probs = self._get_softmax(layer_input)
            return

        # We should predict probabilities of the target outputs. If the target
        # class IDs are not explicitly specified, they are the words at the
        # following time step.
        if self.network.mode.is_target_words():
            target_class_ids = self.network.target_class_ids
        else:
            target_class_ids = self.network.class_input[1:]

        if self._use_nce:
            self.target_probs = self._get_sigmoid(layer_input, target_class_ids)
            # Generate one word for each training word as a negative sample.
            sample_class_ids = self.random.uniform(self.target_probs.shape)
            sample_class_ids *= num_classes
            sample_class_ids = tensor.cast(sample_class_ids, 'int64')
            self.negative_probs = self._get_sigmoid(layer_input,
                                                    sample_class_ids)
            # minibatch_size = self.negative_probs.shape[0] * self.negative_probs.shape[1]
            # minibatch_size = tensor.cast(minibatch_size, theano.config.floatX)
            # cost = -tensor.log(self.target_probs)
            # cost -= minibatch_size * tensor.log(1.0 - self.negative_probs)
            return

        output_probs = self._get_softmax(layer_input)
        if not self.network.mode.is_target_words():
            output_probs = output_probs[:-1]
            target_class_ids = target_class_ids[1:]

        assert_op = tensor.opt.Assert(
            "Mismatch in mini-batch and target classes shape.")
        target_class_ids = assert_op(
            target_class_ids,
            tensor.eq(target_class_ids.shape[0], output_probs.shape[0]))
        target_class_ids = assert_op(
            target_class_ids,
            tensor.eq(target_class_ids.shape[1], output_probs.shape[1]))

        # An index to a flattened input matrix times the vocabulary size can be
        # used to index the same location in the output matrix. The class ID is
        # added to index the probability of that word.
        num_time_steps = output_probs.shape[0]
        num_sequences = output_probs.shape[1]
        output_probs = output_probs.flatten()
        target_class_ids = target_class_ids.flatten()
        minibatch_size = target_class_ids.shape[0]
        output_probs = assert_op(
            output_probs,
            tensor.eq(output_probs.shape[0], minibatch_size * num_classes))
        target_indices = tensor.arange(minibatch_size) * num_classes
        target_indices += target_class_ids
        self.target_probs = output_probs[target_indices]
        self.target_probs.reshape([num_time_steps, num_sequences])

    def _get_softmax(self, layer_input):
        preact = self._tensor_preact(layer_input, 'input')

        # Combine the first two dimensions so that softmax is taken
        # independently for each location, over the output classes. This
        # produces probabilities for the whole vocabulary.
        num_time_steps = preact.shape[0]
        num_sequences = preact.shape[1]
        output_size = preact.shape[2]
        preact = preact.reshape([num_time_steps * num_sequences,
                                 output_size])
        result = tensor.nnet.softmax(preact)
        return result.reshape([num_time_steps, num_sequences, output_size])

    def _get_sigmoid(self, layer_input, target_class_ids):
        weight = self._params['input/W']
        bias = self._params['input/b']

        # Combine the first two dimensions so that sigmoid is taken
        # independently for each preactivation.
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        input_size = layer_input.shape[2]
        layer_input = layer_input.reshape([num_time_steps * num_sequences,
                                           input_size])

        # Create preactivation for only the target outputs.
        target_class_ids = target_class_ids.flatten()
        weight = weight[:, target_class_ids]
        bias = bias[target_class_ids]
        preact = (weight.T * layer_input).sum(1) + bias
        result = tensor.nnet.sigmoid(preact)
        return result.reshape([num_time_steps, num_sequences])
