#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
import numpy

class TextScorer(object):
    """Text Scoring Using a Neural Network Language Model
    """

    def __init__(self, network, profile=False):
        """Creates a Theano function self.costs_function that computes the
        negative log probabilities of given text sequences.

        :type network: RNNLM
        :param network: the neural network object

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        inputs = [network.minibatch_input, network.minibatch_mask]

        # Calculate negative log probability of each word.
        costs = -tensor.log(network.prediction_probs)
        # Apply mask to the costs matrix.
        costs = costs * network.minibatch_mask
        # Sum costs over time steps to get the negative log probability of each
        # sequence.
        outputs = costs.sum(0)

        self.score_function = \
                theano.function(inputs, outputs, profile=profile)

    def negative_log_probability(self, batch_iter):
        """Computes the mean negative log probability of mini-batches read using
        the given iterator.

        ``batch_iter`` is an iterator to the input data. On each call it creates
        a tuple of three 2-dimensional matrices, all indexed by time step and
        sequence. The first matrix contains the word IDs, the second one
        contains class membership probabilities, and the third one masks out
        elements past the sequence ends.

        :type batch_iter: BatchIterator
        :param batch_iter: an iterator that creates mini-batches from the input
                           data

        :rtype: float
        :returns: average sequence negative log probability
        """

        costs = []
        for word_ids, membership_probs, mask in batch_iter:
            # A vector of costs of each sequence in the mini-batch.
            batch_costs = self.score_function(word_ids, mask)
            # A matrix of costs from class membership of each word in the
            # mini-batch.
            membership_costs = -numpy.log(membership_probs)
            membership_costs[mask < 0.5] = 0.0
            batch_costs += membership_costs.sum(0)
            costs.extend(batch_costs)
            if numpy.isnan(numpy.mean(costs)):
                import ipdb; ipdb.set_trace()

        # Return the average sequence cost.
        return numpy.array(costs).mean()

    def score_sentence(self, word_ids, membership_probs):
        """Computes the mean negative log probability of mini-batches read using
        the given iterator.

        ``batch_iter`` is an iterator to the input data. On each call it creates
        a tuple of three 2-dimensional matrices, all indexed by time step and
        sequence. The first matrix contains the word IDs, the second one
        contains class membership probabilities, and the third one masks out
        elements past the sequence ends.

        :type batch_iter: BatchIterator
        :param batch_iter: an iterator that creates mini-batches from the input
                           data

        :rtype: float
        :returns: average sequence negative log probability
        """

        mask = numpy.ones_like(membership_probs)
        # A vector containing the cost of the one sentence.
        costs = self.score_function(word_ids, mask)
        # A matrix of costs from class membership of each word in the one
        # sentence.
        membership_costs = -numpy.log(membership_probs)
        costs += membership_costs.sum(0)
        if numpy.isnan(numpy.mean(costs)):
            import ipdb; ipdb.set_trace()

        # Return the average of the one sentence cost.
        return numpy.array(costs).mean()
