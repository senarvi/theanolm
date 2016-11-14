#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.basiclayer import BasicLayer
from theanolm.debugfunctions import *

class NCELayer(BasicLayer):
    """Output Layer with NCE Support

    Base class for output layers with support for noise-contrastive estimation.
    The base class defines functions for computing unnormalized probabilities.
    """

    def _compute_sample_logprobs(self, layer_input):
        """Creates noise samples for NCE and computes their log probabilities.

        :type layer_input: TensorVariable
        :param layer_input: a 3-dimensional tensor that contains the input
                            vector for each time step in each sequence
        """

        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        num_samples = self.network.num_noise_samples
        num_classes = self.network.vocabulary.num_classes()

        # Sample k noise words from unigram distribution. These are shared
        # across mini-batch. We need to repeat the distribution as many times as
        # we want samples, because multinomial() does not yet use the size
        # argument.
        class_probs = self.network.class_prior_probs
        class_probs = class_probs[None, :]
        class_probs = tensor.tile(class_probs, [num_samples, 1])
        sample = self.network.random.multinomial(pvals=class_probs)
        sample = sample.argmax(1)
        self.shared_sample_logprobs = \
            self._get_target_list_preact(layer_input, sample)
        self.shared_sample = sample

        # Sample k noise words per training word from unigram distribution.
        # multinomial() is only implemented for dimension <= 2, so we'll create
        # a 2-dimensional probability distribution and then reshape the result.
        class_probs = self.network.class_prior_probs
        class_probs = class_probs[None, :]
        num_batch_samples = num_time_steps * num_sequences * num_samples
        class_probs = tensor.tile(class_probs, [num_batch_samples, 1])
        sample = self.network.random.multinomial(pvals=class_probs)
        sample = sample.argmax(1)
        sample = sample.reshape([num_time_steps, num_sequences, num_samples])
        self.sample_logprobs = \
            self._get_target_preact(layer_input, sample)
        self.sample = sample

    def _compute_unnormalized_logprobs(self, layer_input):
        """Computes unnormalized output (preactivations).

        :type layer_input: TensorVariable
        :param layer_input: a 3-dimensional tensor that contains the input
                            vector for each time step in each sequence

        Unnormalized probabilities and noise probabilities are used by noise-
        contrastive estimation. Softmax output is exponential, so the
        preactivations can be seen as unnormalized log probabilities.
        """

        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        target_class_ids = self.network.target_class_ids[:, :, None]
        result = self._get_target_preact(layer_input, target_class_ids)
        self.unnormalized_logprobs = result.reshape([num_time_steps,
                                                     num_sequences])

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

        # Create preactivation for only the target outputs.
        weight = self._params[self._param_path('input/W')]
        bias = self._params[self._param_path('input/b')]
        target_class_ids = target_class_ids.flatten()
        weight = weight.T
        weight = weight[target_class_ids, :]
        weight = weight.reshape([minibatch_size, -1, input_size])
        bias = bias[target_class_ids]
        bias = bias.reshape([minibatch_size, -1])

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
