#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
import numpy
from theanolm.matrixfunctions import test_value
from theanolm.exceptions import NumberError

class TextScorer(object):
    """Text Scoring Using a Neural Network Language Model
    """

    def __init__(self, network, ignore_unk=False, unk_penalty=None,
                 profile=False):
        """Creates two Theano function, ``self._target_logprobs_function()``,
        which computes the log probabilities predicted by the neural network for
        the words in a mini-batch, and ``self._total_logprob_function()``, which
        returns the total log probability.

        Both functions take as arguments four matrices:
        1. Word IDs in the shape of a mini-batch. The functions will only use
           the input words (not the last time step).
        2. Class IDs in the shape of a mini-batch. The functions will slice this
           into input and output.
        3. Class membership probabilities in the shape of a mini-batch, but only
           for the output words (not the first time step).
        4. Mask in the shape of a mini-batch, but only for the output words (not
           for the first time step).

        ``self._target_logprobs_function()`` will return a matrix of predicted
        log probabilities for the output words (excluding the first time step)
        and the mask after possibly applying special UNK handling.
        ``self._total_logprob_function()`` will return the total log probability
        of the predicted (unmasked) words and the number of those words.

        :type network: Network
        :param network: the neural network object

        :type ignore_unk: bool
        :param ignore_unk: if set to True, <unk> tokens are excluded from
                           perplexity computation

        :type unk_penalty: float
        :param unk_penalty: if set to othern than None, used as <unk> token
                            score

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self._ignore_unk = ignore_unk
        self._unk_penalty = unk_penalty
        self._vocabulary = network.vocabulary
        self._unk_id = network.vocabulary.word_to_id['<unk>']

        # The functions take as input a mini-batch of word IDs and class IDs,
        # and slice input and target IDs for the network.
        batch_word_ids = tensor.matrix('textscorer/batch_word_ids',
                                       dtype='int64')
        batch_word_ids.tag.test_value = test_value(
            size=(101, 16), high=self._vocabulary.num_words())
        batch_class_ids = tensor.matrix('textscorer/batch_class_ids',
                                        dtype='int64')
        batch_class_ids.tag.test_value = test_value(
            size=(101, 16), high=self._vocabulary.num_classes())
        membership_probs = tensor.matrix('textscorer/membership_probs',
                                         dtype=theano.config.floatX)
        membership_probs.tag.test_value = test_value(
            size=(100, 16), high=1.0)

        logprobs = tensor.log(network.target_probs())
        # Add logprobs from the class membership of the predicted word at each
        # time step of each sequence.
        logprobs += tensor.log(membership_probs)
        # If requested, predict <unk> with constant score.
        target_word_ids = batch_word_ids[1:]
        if not self._unk_penalty is None:
            unk_mask = tensor.eq(target_word_ids, self._unk_id)
            unk_indices = unk_mask.nonzero()
            logprobs = tensor.set_subtensor(logprobs[unk_indices],
                                            self._unk_penalty)
        # Ignore logprobs predicting a word that is past the sequence end, and
        # possibly also those that are predicting <unk> token.
        mask = network.mask
        if self._ignore_unk:
            mask *= tensor.neq(target_word_ids, self._unk_id)
        logprobs *= tensor.cast(mask, theano.config.floatX)

        # Ignore unused input variables, because is_training is only used by
        # dropout layer.
        self._target_logprobs_function = theano.function(
            [batch_word_ids, batch_class_ids, membership_probs, network.mask],
            [logprobs, mask],
            givens=[(network.input_word_ids, batch_word_ids[:-1]),
                    (network.input_class_ids, batch_class_ids[:-1]),
                    (network.target_class_ids, batch_class_ids[1:]),
                    (network.is_training, numpy.int8(0))],
            name='target_logprobs',
            on_unused_input='ignore',
            profile=profile)
        self._total_logprob_function = theano.function(
            [batch_word_ids, batch_class_ids, membership_probs, network.mask],
            [logprobs.sum(), mask.sum()],
            givens=[(network.input_word_ids, batch_word_ids[:-1]),
                    (network.input_class_ids, batch_class_ids[:-1]),
                    (network.target_class_ids, batch_class_ids[1:]),
                    (network.is_training, numpy.int8(0))],
            name='total_logprob',
            on_unused_input='ignore',
            profile=profile)

    def score_batch(self, word_ids, class_ids, membership_probs, mask):
        """Computes the log probabilities predicted by the neural network for
        the words in a mini-batch.

        Indices in the resulting list of lists will be a transpose of those of
        the input matrices matrices, so that the first index is the sequence,
        not the time step.

        :type word_ids: numpy.ndarray of an integer type
        :param word_ids: a 2-dimensional matrix, indexed by time step and
                         sequence, that contains the word IDs

        :type class_ids: numpy.ndarray of an integer type
        :param class_ids: a 2-dimensional matrix, indexed by time step and
                          sequence, that contains the class IDs

        :type membership_probs: numpy.ndarray of a floating point type
        :param membership_probs: a 2-dimensional matrix, indexed by time step
                                 and sequences, that contains the class
                                 membership probabilities of the words

        :type mask: numpy.ndarray of a floating point type
        :param mask: a 2-dimensional matrix, indexed by time step and sequence,
                     that masks out elements past the sequence ends

        :rtype: list of lists
        :returns: logprob of each word in each sequence
        """

        result = []
        membership_probs = membership_probs.astype(theano.config.floatX)

        # target_logprobs_function() uses the word and class IDs of the entire
        # mini-batch, but membership probs and mask are only for the output.
        logprobs, mask = \
            self._target_logprobs_function(word_ids,
                                           class_ids,
                                           membership_probs[1:],
                                           mask[1:])
        for seq_index in range(logprobs.shape[1]):
            seq_logprobs = logprobs[:,seq_index]
            seq_mask = mask[:,seq_index]
            seq_logprobs = seq_logprobs[seq_mask == 1]
            result.append(seq_logprobs)

        return result

    def compute_perplexity(self, batch_iter):
        """Computes the perplexity of text read using the given iterator.

        ``batch_iter`` is an iterator to the input data. On each call it creates
        a two 2-dimensional matrices, both indexed by time step and sequence.
        The first matrix contains the word IDs, the second one masks out
        elements past the sequence ends.

        :type batch_iter: BatchIterator
        :param batch_iter: an iterator that creates mini-batches from the input
                           data

        :rtype: float
        :returns: perplexity, i.e. exponent of negative log probability
                  normalized by the number of words
        """

        logprob = 0
        num_words = 0

        for word_ids, _, mask in batch_iter:
            class_ids, membership_probs = \
                self._vocabulary.get_class_memberships(word_ids)
            membership_probs = membership_probs.astype(theano.config.floatX)

            # total_logprob_function() uses the word and class IDs of the entire
            # mini-batch, but membership probs and mask are only for the output.
            batch_logprob, batch_num_words = \
                self._total_logprob_function(word_ids,
                                             class_ids,
                                             membership_probs[1:],
                                             mask[1:])
            if numpy.isnan(batch_logprob):
                raise NumberError("Log probability of a mini-batch is NaN.")
            if numpy.isinf(batch_logprob):
                raise NumberError("Log probability of a mini-batch is +/- infinity.")

            logprob += batch_logprob
            num_words += batch_num_words

        if num_words == 0:
            raise ValueError("Zero words for computing perplexity. Does the "
                             "evaluation data contain only OOV words?")
        cross_entropy = -logprob / num_words
        return numpy.exp(cross_entropy)

    def score_sequence(self, word_ids, class_ids, membership_probs):
        """Computes the log probability of a word sequence.

        :type word_ids: ndarray
        :param word_ids: a vector of word IDs

        :type class_ids: list of ints
        :param class_ids: corresponding class IDs

        :type membership_probs: list of floats
        :param membership_probs: list of class membership probabilities

        :rtype: float
        :returns: log probability of the word sequence
        """

        # Create 2-dimensional matrices representing the transposes of the
        # vectors.
        word_ids = numpy.transpose(word_ids[numpy.newaxis])
        class_ids = numpy.array([[x] for x in class_ids], numpy.int64)
        membership_probs = numpy.array(
            [[x] for x in membership_probs]).astype(theano.config.floatX)
        # Mask used by the network is all ones.
        mask = numpy.ones(word_ids.shape, numpy.int8)

        # total_logprob_function() uses the word and class IDs of the entire
        # mini-batch, but membership probs and mask are only for the output.
        logprob, _ = self._total_logprob_function(word_ids,
                                                  class_ids,
                                                  membership_probs[1:],
                                                  mask[1:])
        if numpy.isnan(logprob):
            raise NumberError("Log probability of a sequence is NaN.")
        if numpy.isinf(logprob):
            raise NumberError("Log probability of a sequence is +/- infinity.")

        return logprob

    def unk_ignored(self):
        """Indicates whether the scorer ignores <unk> tokens.

        :rtype: bool
        :returns: ``True`` if the scorer ignores <unk> tokens, ``False``
                  otherwise.
        """

        return self._ignore_unk
