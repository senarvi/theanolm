#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
import numpy
from theanolm.exceptions import NumberError

class TextScorer(object):
    """Text Scoring Using a Neural Network Language Model
    """

    def __init__(self, network, classes_to_ignore=[], profile=False):
        """Creates a Theano function self.score_function that computes the
        log probabilities predicted by the neural network for the words in a
        mini-batch.

        self.score_function takes as arguments two matrices, the input word IDs
        and mask, and returns a matrix of word prediction log probabilities. The
        matrices are indexed by time step and word sequence, output containing
        one less time step, since the last time step is not predicting any word.

        :type network: Network
        :param network: the neural network object

        :type classes_to_ignore: list of ints
        :param classes_to_ignore: list of class IDs that will be ignored when
                                  computing the cost

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.classes_to_ignore = classes_to_ignore

        inputs = [network.input, network.mask]
        logprobs = tensor.log(network.prediction_probs)
        # Ignore unused input variables, because is_training is only used by
        # dropout layer.
        self.score_function = theano.function(
            inputs,
            logprobs,
            givens=[(network.is_training, numpy.int8(0))],
            name='text_scorer',
            on_unused_input='ignore',
            profile=profile)

    def score_batch(self, word_ids, membership_probs, mask):
        """Computes the log probabilities predicted by the neural network for
        the words in a mini-batch.

        :type word_ids: numpy.ndarray of an integer type
        :param word_ids: a 2-dimensional matrix, indexed by time step and
                         sequence, that contains the word IDs

        :type membership_probs: numpy.ndarray of a floating point type
        :param membership_probs: a 2-dimensional matrix, indexed by time step
                                 and sequences, that contains the class
                                 membership probabilities of the words

        :type mask: numpy.ndarray of a floating point type
        :param mask: a 2-dimensional matrix, indexed by time step and sequence,
                     that masks out elements past the sequence ends.

        :rtype: list of lists
        :returns: logprob of each word in each sequence
        """

        result = []

        # A matrix of neural network logprobs of each word in each sequence.
        logprobs = self.score_function(word_ids, mask)
        # Add logprobs from the class membership of the predicted word at each
        # time step of each sequence.
        logprobs += numpy.log(membership_probs[1:])
        # Ignore logprobs predicting a word that is past the sequence end or one
        # of the words to be ignored.
        for word_id in self.classes_to_ignore:
            mask[word_ids == word_id] = 0
        for seq_index in range(logprobs.shape[1]):
            seq_logprobs = logprobs[:,seq_index]
            seq_mask = mask[1:,seq_index]
            seq_logprobs = seq_logprobs[seq_mask == 1]
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

    def score_sequence(self, word_ids, membership_probs):
        """Computes the log probability of a word sequence.

        :type word_ids: list of ints
        :param word_ids: list of word IDs

        :type membership_probs: list of floats
        :param membership_probs: list of class membership probabilities

        :rtype: float
        :returns: log probability of the sentence
        """

        # Create 2-dimensional matrices representing the transposes of the
        # vectors.
        word_ids = numpy.array([[x] for x in word_ids], numpy.int64)
        membership_probs = numpy.array([[x] for x in membership_probs]).astype(
            theano.config.floatX)
        # Mask used by the network is all ones.
        mask = numpy.ones(word_ids.shape, numpy.int8)

        logprobs = self.score_function(word_ids, mask)
        # Add logprobs from the class membership of the predicted word at each
        # time step of each sequence.
        logprobs += numpy.log(membership_probs[1:])
        # Zero out logprobs predicting a word to be ignored. Numpy preserves the
        # data type when multiplying by an int8.
        for word_id in self.classes_to_ignore:
            mask[word_ids == word_id] = 0
        logprobs *= mask[1:]
        logprob = logprobs.sum()

        if numpy.isnan(logprob):
            raise NumberError("Sentence logprob has NaN value.")
        return logprob
