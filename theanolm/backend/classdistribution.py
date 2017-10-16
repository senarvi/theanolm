#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements classes for sampling noise words.
"""

from abc import abstractmethod, ABCMeta

import numpy
import theano
import theano.tensor as tensor


class ClassDistribution(object, metaclass=ABCMeta):
    """Base Class for Probability Distributions

    A probability distribution class implements methods for sampling words and
    converting words to probabilities.
    """

    def __init__(self, random):
        """Constructs a probability distribution.

        :type random: MRG_RandomStreams
        :param random: a random number generator
        """

        self._random = random
        self._float_type = numpy.dtype(theano.config.floatX).type

    @abstractmethod
    def sample(self, minibatch_size, num_samples):
        """Samples given number of class IDs per mini-batch element.
        """

        assert False

    @abstractmethod
    def probs(self, class_ids):
        """Converts class IDs to probabilities.

        The returned value may be a NumPy array or Theano tensor. It may be an
        array of the same shape as ``class_ids`` or a scalar which can be
        broadcasted to that shape.

        :type class_ids: numpy.ndarray
        :param class_ids: classes whose probabilities are requested

        :rtype: theano.config.floatX, numpy.ndarray, or Variable
        :returns: an array of the same shape as ``class_ids`` or a scalar which
                  can be broadcasted to that shape
        """

        assert False


class UniformDistribution(ClassDistribution):
    def __init__(self, random, support):
        """Constructs a uniform probability distribution.

        :type random: MRG_RandomStreams
        :param random: a random number generator

        :type support: int
        :param support: the sampled values will be in the range from 0 to
                        ``support - 1``
        """

        super().__init__(random)
        self._support = support

    def sample(self, minibatch_size, num_samples):
        """Samples given number of class IDs per mini-batch element.

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

    def probs(self, class_ids):
        """Converts class IDs to probabilities.

        The returned value is a scalar. It will be broadcasted to the correct
        shape by Theano.

        :type class_ids: Variable
        :param class_ids: a symbolic vector describing the classes whose
                          probabilities are requested (ignored)

        :rtype: theano.config.floatX
        :returns: the probability of a single class
        """

        return self._float_type(1.0 / self._support)


class LogUniformDistribution(ClassDistribution):
    def __init__(self, random, support):
        """Constructs a probability distribution such that the logarithm of the
        samples is uniformly distributed.

        :type random: MRG_RandomStreams
        :param random: a random number generator

        :type support: int
        :param support: the sampled values will be in the range from 0 to
                        ``support - 1``
        """

        super().__init__(random)
        # Random numbers will be in the range [0, log(support + 1)[.
        self._log_support = numpy.log(support + 1)

    def sample(self, minibatch_size, num_samples):
        """Samples given number of class IDs per mini-batch element.

        :type minibatch_size: int
        :param minibatch_size: number of mini-batch elements

        :type num_samples: int
        :param num_samples: number of samples to draw for each mini-batch element

        :rtype: Variable
        :returns: a 2-dimensional tensor variable describing the ``num_samples``
                  random samples for ``minibatch_size`` mini-batch element
        """

        logs = self._random.uniform(size=(minibatch_size, num_samples),
                                    high=self._log_support)
        # The exponent will be in the range [1, support + 1[.
        sample = tensor.exp(logs) - 1
        return sample.astype('int64')

    def probs(self, class_ids):
        """Converts class IDs to probabilities.

        The returned value is a scalar. It will be broadcasted to the correct
        shape by Theano.

        :type class_ids: Variable
        :param class_ids: a symbolic vector describing the classes whose
                          probabilities are requested

        :rtype: Variable
        :returns: probabilities of the classes
        """

        # A sample will be in the range [x, x + 1[ when the log is in the range
        # [log(x + 1), log(x + 2)[. Thus the probability of x is
        # (log(x + 2) - log(x + 1)) / log_support.
        range = (class_ids + 2) / (class_ids + 1)
        return tensor.log(range) / self._log_support

class MultinomialDistribution(ClassDistribution):
    def __init__(self, random, probs):
        """Constructs a multinomial probability distribution.

        :type random: MRG_RandomStreams
        :param random: a random number generator

        :type probs: Variable
        :param probs: a tensor vector that defines the distribution where to
                      sample from
        """

        super().__init__(random)
        self._probs = probs

    def sample(self, minibatch_size, num_samples):
        """Samples given number of class IDs per mini-batch element.

        Theano supports currently only sampling without replacement. Thus if a
        different set of samples is required for each mini-batch element, we
        repeat the distribution for each mini-batch element.

        At some point the old interface, ``multinomial_wo_replacement()`` was
        faster, but is not supported anymore. In the future ``target='cpu'`` may
        improve the speed.

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
        sample = self._random.choice(size=num_samples, replace=False, p=probs,
                                     target='cpu')
        # Some versions of Theano seem to return crazy high or low numbers because
        # of some rounding errors, so we take the modulo to be safe.
        sample %= probs.shape[1]
        return sample

    def probs(self, class_ids):
        """Converts class IDs to probabilities.

        The returned value is a scalar. It will be broadcasted to the correct
        shape by Theano.

        :type class_ids: Variable
        :param class_ids: a symbolic vector describing the classes whose
                          probabilities are requested

        :rtype: Variable
        :returns: symbolic vector describing the probabilities of the classes
        """

        return self._probs[class_ids]
