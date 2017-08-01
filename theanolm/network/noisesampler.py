#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements classes for sampling noise words.
"""

import numpy
import theano.tensor as tensor


class MultinomialSampler(object):
    def __init__(self, random, probs):
        """Constructs a random sampler that samples from multinomial
        distribution.

        :type random: MRG_RandomStreams
        :param random: a random number generator

        :type probs: Variable
        :param probs: a tensor vector that defines the distribution where to
                      sample from
        """

        self._random = random
        self._probs = probs

    def sample(self, minibatch_size, num_samples):
        """Samples given number of word IDs per mini-batch element.

        Theano supports currently only sampling without replacement. Thus if a
        different set of samples is required for each mini-batch element, we
        repeat the distribution for each mini-batch element.

        At some point the old interface, ``multinomial_wo_replacement()`` was
        faster, but is not supported anymore.

        :type minibatch_size: int
        :param minibatch_size: number of mini-batch elements

        :type num_samples: int
        :param num_samples: number of samples to draw for each mini-batch element

        :rtype: Variable
        :returns: a 2-dimensional tensor variable describing the ``num_samples``
                  random samples for ``minibatch_size`` mini-batch element
        """

        probs = self._probs[None, :]
        probs = tensor.tile(probs, [minibatch_size, 1])
#        return self._random.multinomial_wo_replacement(pvals=probs, n=num_samples)
        sample = self._random.choice(size=num_samples, replace=False, p=probs)
        # Some versions of Theano seem to return crazy high or low numbers because
        # of some rounding errors, so we take the modulo to be safe.
        sample %= probs.shape[1]
        return sample


class LogUniformSampler(object):
    def __init__(self, random, support):
        """Constructs a random sampler that samples random integers whose
        logarithm is uniformly distributed.

        :type random: MRG_RandomStreams
        :param random: a random number generator

        :type support: int
        :param support: the sampled values will be in the range from 0 to
                        ``support - 1``
        """

        self._random = random
        # Random numbers will be in the range [0, log(support + 1)[.
        self._log_support = numpy.log(support + 1)

    def sample(self, minibatch_size, num_samples):
        """Samples given number of word IDs per mini-batch element.

        :type minibatch_size: int
        :param minibatch_size: number of mini-batch elements

        :type num_samples: int
        :param num_samples: number of samples to draw for each mini-batch element

        :rtype: Variable
        :returns: a 2-dimensional tensor variable describing the ``num_samples``
                  random samples for ``minibatch_size`` mini-batch element
        """

        print(minibatch_size, num_samples, self._log_support)
        logs = self._random.uniform(size=(minibatch_size, num_samples),
                                    high=self._log_support)
        # The exponent will be in the range [1, support + 1[.
        sample = tensor.exp(logs) - 1
        return sample.astype('int64')


class UniformSampler(object):
    def __init__(self, random, support):
        """Constructs a random sampler that samples from uniform distribution.

        :type random: MRG_RandomStreams
        :param random: a random number generator

        :type support: int
        :param support: the sampled values will be in the range from 0 to
                        ``support - 1``
        """

        self._random = random
        self._support = support

    def sample(self, minibatch_size, num_samples):
        """Samples given number of word IDs per mini-batch element.

        :type minibatch_size: int
        :param minibatch_size: number of mini-batch elements

        :type num_samples: int
        :param num_samples: number of samples to draw for each mini-batch element

        :rtype: Variable
        :returns: a 2-dimensional tensor variable describing the ``num_samples``
                  random samples for ``minibatch_size`` mini-batch element
        """

        sample = self._random.uniform(size=(minibatch_size, num_samples),
                                      high=self._support)
        return sample.astype('int64')
