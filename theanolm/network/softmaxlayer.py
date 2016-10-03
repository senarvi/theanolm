#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.basiclayer import BasicLayer
from theanolm.debugfunctions import *

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

        self.output_probs = self._get_softmax(layer_input)
        if self.network.target_class_ids is None:
            self.target_probs = None
            self.unnormalized_logprobs = None
            self.sampled_logprobs = None
            return

        output_probs = self.output_probs
        target_class_ids = self.network.target_class_ids
        num_time_steps = output_probs.shape[0]
        num_sequences = output_probs.shape[1]

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

        # Unnormalized probabilities and noise probabilities are used by noise-
        # contrastive estimation. Softmax output is exponential, so the
        # preactivations can be seen as unnormalized log probabilities.
        self.unnormalized_logprobs = \
            self._get_target_preact(layer_input, target_class_ids[:,:,None])
        self.unnormalized_logprobs = \
            self.unnormalized_logprobs.reshape([num_time_steps, num_sequences])

        # Sample k noise words from uniform distribution. These are shared
        # across mini-batch.
        num_samples = self.network.num_noise_samples
        shared_sample = self.network.random.uniform((num_samples,))
        shared_sample *= num_classes
        shared_sample = tensor.cast(shared_sample, 'int64')
        self.shared_sample_logprobs = \
            self._get_target_list_preact(layer_input, shared_sample)

        # Sample k noise words per training word from uniform distribution.
        sample = self.network.random.uniform((num_time_steps,
                                              num_sequences,
                                              num_samples))
        sample *= num_classes
        sample = tensor.cast(sample, 'int64')
        self.sample_logprobs = \
            self._get_target_preact(layer_input, sample)

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

    def _get_softmax(self, layer_input):
        """Structures the softmax output for every class.

        :type layer_input: TensorVariable
        :param layer_input: a 3-dimensional tensor that contains the input
                            vector for each time step in each sequence

        :rtype: TensorVariable
        :returns: a 3-dimensional tensor that contains the softmax output for
                  every class, at each time step of each sequence
        """

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

    def _get_target_preact(self, layer_input, target_class_ids):
        """Structures the preactivations for given targets. One target is given
        for each word in the minibatch.

        :type layer_input: TensorVariable
        :param layer_input: a 3-dimensional tensor that contains the input
                            vector for each time step in each sequence

        :type target_class_ids: TensorVariable
        :param target_class_ids: a 3-dimensional tensor that contains one or more
                                 target class IDs for each time step in each
                                 sequence

        :rtype: TensorVariable
        :returns: a 2-dimensional tensor that contains the preactivation of the
                  target word, for each time step in each sequence
        """

        # Combine the first two dimensions so that weight matrix will have equal
        # dimensions and element-wise multiplication is possible.
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        input_size = layer_input.shape[2]
        minibatch_size = num_time_steps * num_sequences
        layer_input = layer_input.reshape([minibatch_size, 1, input_size])

        layer_input = assert_tensor_eq(
            layer_input,
            'target_class_ids.shape[0]',
            'num_time_steps',
            target_class_ids.shape[0],
            num_time_steps)
        layer_input = assert_tensor_eq(
            layer_input,
            'target_class_ids.shape[1]',
            'num_sequences',
            target_class_ids.shape[1],
            num_sequences)
#        layer_input = assert_tensor_eq(
#            layer_input,
#            'target_class_ids.shape[2]',
#            'self.network.num_noise_samples',
#            target_class_ids.shape[2],
#            self.network.num_noise_samples)

        # Create preactivation for only the target outputs.
        weight = self._params[self._param_path('input/W')]
        bias = self._params[self._param_path('input/b')]
        target_class_ids = target_class_ids.flatten()
        weight = weight.T
        weight = weight[target_class_ids, :]
        weight = weight.reshape([minibatch_size, -1, input_size])
        bias = bias[target_class_ids]
        bias = bias.reshape([minibatch_size, -1])

 #       layer_input = assert_tensor_eq(
 #           layer_input,
 #           'weight.shape[1]',
 #           'self.network.num_noise_samples',
 #           weight.shape[1],
 #           self.network.num_noise_samples)
 #       layer_input = assert_tensor_eq(
 #           layer_input,
 #           'bias.shape[1]',
 #           'self.network.num_noise_samples',
 #           bias.shape[1],
 #           self.network.num_noise_samples)

        result = (layer_input * weight).sum(2) + bias
        return result.reshape([num_time_steps, num_sequences, -1])

    def _get_target_list_preact(self, layer_input, target_class_ids):
        """Structures the preactivations for a list of target classes.
        Preactivations at each word are computed for all the targets.

        :type layer_input: TensorVariable
        :param layer_input: a 3-dimensional tensor that contains the input
                            vector for each time step in each sequence

        :type target_class_ids: TensorVariable
        :param target_class_ids: a list of target classes

        :rtype: TensorVariable
        :returns: a 3-dimensional tensor that contains the preactivation for
                  every target word, at each time step of each sequence
        """

        assert_op = tensor.opt.Assert(
            "A list of target IDs required, but a multidimensional tensor was "
            "given.")
        target_class_ids = assert_op(
            target_class_ids,
            tensor.eq(target_class_ids.ndim, 1))

        # Create preactivation for only the target outputs.
        weight = self._params[self._param_path('input/W')]
        bias = self._params[self._param_path('input/b')]
        weight = weight[:, target_class_ids]
        bias = bias[target_class_ids]
        return tensor.dot(layer_input, weight) + bias
