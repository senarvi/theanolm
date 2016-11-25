#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theanolm.network.basiclayer import BasicLayer

class SamplingOutputLayer(BasicLayer):
    """Sampling Support for Output Layer

    Base class for output layers with support for sampling noise words and
    computing unnormalized probabilities. This is needed for Noise-contrastive
    estimation and BlackOut.
    """

    def _compute_sample_logprobs(self, layer_input):
        """Creates noise samples for NCE and BlackOut, and computes their log
        probabilities.

        :type layer_input: TensorVariable
        :param layer_input: a 3-dimensional tensor that contains the input
                            vector for each time step in each sequence
        """

        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        num_samples = self.network.num_noise_samples
        num_classes = numpy.int64(self.network.vocabulary.num_classes())
        random = self.network.random

        # Sample k noise words from unigram distribution. These are shared
        # across mini-batch. We need to repeat the distribution as many times as
        # we want samples, because multinomial() does not yet use the size
        # argument.
        if self.network.noise_probs is None:
            # The upper bound is exclusive, so this always creates samples that
            # are < num_classes
            sample = random.uniform((num_samples,)) * num_classes
            sample = sample.astype('int64')
        else:
            class_probs = self.network.noise_probs[None, :]
            sample = random.multinomial_wo_replacement(pvals=class_probs,
                                                       n=num_samples)
            sample = sample[0, :]
            # For some reason (maybe a rounding error) it may happen that the
            # sample contains a very high or negative value.
            sample = tensor.maximum(sample, 0)
            sample = tensor.minimum(sample, num_classes - 1)
        self.shared_sample_logprobs = \
            self._get_target_list_preact(layer_input, sample)
        self.shared_sample = sample

        # Sample k noise words per training word from unigram distribution.
        # multinomial() is only implemented for dimension <= 2, so we'll create
        # a 2-dimensional probability distribution and then reshape the result.
        minibatch_size = num_time_steps * num_sequences
        if self.network.noise_probs is None:
            # The upper bound is exclusive, so this always creates samples that
            # are < num_classes
            num_batch_samples = minibatch_size * num_samples
            sample = random.uniform((num_batch_samples,)) * num_classes
            sample = sample.astype('int64')
        else:
            class_probs = self.network.noise_probs[None, :]
            class_probs = tensor.tile(class_probs, [minibatch_size, 1])
            # Since we sample different noise words for different data words, we
            # could set the probability of the correct data words to zero, as
            # suggested in the BlackOut paper, but this seems to make the
            # results very bad.
#            target_class_ids = self.network.target_class_ids.flatten()
#            target_sample_ids = tensor.arange(minibatch_size)
#            class_probs = tensor.set_subtensor(
#                class_probs[(target_sample_ids, target_class_ids)], 0)
#            class_probs /= class_probs.sum()
            sample = random.multinomial_wo_replacement(pvals=class_probs,
                                                       n=num_samples)
            # For some reason (maybe a rounding error) it may happen that the
            # sample contains a very high or negative value.
            sample = tensor.maximum(sample, 0)
            sample = tensor.minimum(sample, num_classes - 1)

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
        """Constructs the preactivations for given targets. One or more target
        outputs are given for each element in the mini-batch.

        :type layer_input: TensorVariable
        :param layer_input: a 3-dimensional tensor that contains the input
                            vector for each time step in each sequence

        :type target_class_ids: TensorVariable
        :param target_class_ids: a 3-dimensional tensor that contains one or
                                 more target class IDs for each time step in
                                 each sequence

        :rtype: TensorVariable
        :returns: a 3-dimensional tensor that contains the preactivation for
                  each target class, for each time step in each sequence
        """

        weight = self._params[self._param_path('input/W')]
        bias = self._params[self._param_path('input/b')]
        weight = weight.T
        weight = weight[target_class_ids, :]
        # The old GPU backend does not implement GpuAdvancedIncSubtensor1_dev20
        # for vectors, which is why the very slow GpuAdvancedIncSubtensor1 will
        # be selected if we index a vector.
        bias = bias[:, None]
        bias = bias[target_class_ids, 0]
#        bias = bias[target_class_ids]
        return (layer_input[:, :, None, :] * weight).sum(3) + bias

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

        weight = self._params[self._param_path('input/W')]
        bias = self._params[self._param_path('input/b')]
        weight = weight[:, target_class_ids]
        # The old GPU backend does not implement GpuAdvancedIncSubtensor1_dev20
        # for vectors, which is why the very slow GpuAdvancedIncSubtensor1 will
        # be selected if we index a vector.
        bias = bias[:, None]
        bias = bias[target_class_ids, 0]
#        bias = bias[target_class_ids]
        return tensor.dot(layer_input, weight) + bias
