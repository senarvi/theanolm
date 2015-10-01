#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
import numpy

class TextScorer(object):
    """Text Scoring Using a Neural Network Language Model
    """

    def __init__(self, network, profile=False):
        """Creates a Theano function self.score_function that computes the
        log probabilities of a mini-batch.

        :type network: Network
        :param network: the neural network object

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        inputs = [network.minibatch_input, network.minibatch_mask]
        logprobs = tensor.log(network.prediction_probs)
        logprobs = logprobs * network.minibatch_mask
        self.score_function = theano.function(inputs, logprobs, profile=profile)

    def score_batch(self, word_ids, membership_probs, mask):
        """Computes the log probability of each word in a mini-batch.

        :type word_ids: numpy.ndarray of int64s
        :param word_ids: a 2-dimensional matrix, indexed by time step and
                         sequence, that contains the word IDs

        :type membership_probs: numpy.ndarray of float32s
        :param membership_probs: a 2-dimensional matrix, indexed by time step
                                 and sequences, that contains the class
                                 membership probabilities of the words

        :type mask: numpy.ndarray of float32s
        :param mask: a 2-dimensional matrix, indexed by time step and sequence,
                     that masks out elements past the sequence ends.

        :rtype: list of lists
        :returns: logprob of each word in each sequence
        """

        result = []

        # A matrix of neural network logprobs of each word in each sequence.
        logprobs = self.score_function(word_ids, mask)
        # Add logprobs from class membership of each word in each sequence.
        logprobs += numpy.log(membership_probs)
        for seq_index in range(logprobs.shape[1]):
            seq_logprobs = logprobs[:,seq_index]
            seq_mask = mask[:,seq_index]
            seq_logprobs = [lp for lp, m in zip(seq_logprobs, seq_mask)
                            if m >= 0.5]
            if numpy.isnan(sum(seq_logprobs)):
                raise NumberError("Sequence logprob has NaN value.")
            result.append(seq_logprobs)

        return result

    def compute_perplexity(self, batch_iter):
        """Computes the perplexity of text read using the given iterator.

        ``batch_iter`` is an iterator to the input data. On each call it creates
        a tuple of three 2-dimensional matrices, all indexed by time step and
        sequence. The first matrix contains the word IDs, the second one
        contains class membership probabilities, and the third one masks out
        elements past the sequence ends.

        :type batch_iter: BatchIterator
        :param batch_iter: an iterator that creates mini-batches from the input
                           data

        :rtype: float
        :returns: perplexity, i.e. exponent of negative log probability
                  normalized by the number of words
        """

        total_logprob = 0
        num_words = 0

        for word_ids, membership_probs, mask in batch_iter:
            logprobs = self.score_batch(word_ids, membership_probs, mask)
            for seq_logprobs in logprobs:
                total_logprob += sum(seq_logprobs)
                num_words += len(seq_logprobs)
        cross_entropy = -total_logprob / num_words
        return numpy.exp(cross_entropy)

    def score_sentence(self, word_ids, membership_probs):
        """Computes the log probability of a sentence.

        :type word_ids: numpy.ndarray of int64s
        :param word_ids: a 2-dimensional matrix representing a transposed
        vector of word IDs

        :type membership_probs: numpy.ndarray of float32s
        :param membership_probs: a 2-dimensional matrix representing a
        transposed vector of class membership probabilities

        :rtype: float
        :returns: log probability of the sentence
        """

        mask = numpy.ones_like(membership_probs)
        logprob = self.score_function(word_ids, mask).sum()
        # Add the logprob of class membership of each word.
        logprob += numpy.log(membership_probs).sum()
        if numpy.isnan(logprob):
            raise NumberError("Sentence logprob has NaN value.")
        return logprob
