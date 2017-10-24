#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the TextScorer class.
"""

import logging

import numpy
import theano
import theano.tensor as tensor

from theanolm.backend import NumberError
from theanolm.backend import test_value
from theanolm.parsing import utterance_from_line

class TextScorer(object):
    """Text Scoring Using a Neural Network Language Model
    """

    def __init__(self, network, use_shortlist=True, exclude_unk=False,
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
        and the mask. ``<unk>`` tokens are also masked out if ``exclude_unk`` is
        set to ``True``. ``self._total_logprob_function()`` will return the
        total log probability of the predicted (unmasked) words and the number
        of those words.

        :type network: Network
        :param network: the neural network object

        :type use_shortlist: bool
        :param use_shortlist: if ``True``, the ``<unk>`` probability is
                              distributed among the out-of-shortlist words

        :type exclude_unk: bool
        :param exclude_unk: if set to ``True``, ``<unk>`` tokens are excluded
                            from probability computation

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self._vocabulary = network.vocabulary
        self._unk_id = self._vocabulary.word_to_id['<unk>']

        # The functions take as input a mini-batch of word IDs and class IDs,
        # and slice input and target IDs for the network.
        batch_word_ids = tensor.matrix('textscorer/batch_word_ids',
                                       dtype='int64')
        batch_word_ids.tag.test_value = test_value(
            size=(21, 4), high=self._vocabulary.num_words())
        batch_class_ids = tensor.matrix('textscorer/batch_class_ids',
                                        dtype='int64')
        batch_class_ids.tag.test_value = test_value(
            size=(21, 4), high=self._vocabulary.num_classes())
        membership_probs = tensor.matrix('textscorer/membership_probs',
                                         dtype=theano.config.floatX)
        membership_probs.tag.test_value = test_value(
            size=(20, 4), high=1.0)

        # Convert out-of-shortlist words to <unk> in input.
        shortlist_size = self._vocabulary.num_shortlist_words()
        input_word_ids = batch_word_ids[:-1]
        oos_indices = tensor.ge(input_word_ids, shortlist_size).nonzero()
        input_word_ids = tensor.set_subtensor(input_word_ids[oos_indices],
                                              self._unk_id)
        # Out-of-shortlist words are already in <unk> class, because they don't
        # have own classes.
        input_class_ids = batch_class_ids[:-1]
        target_class_ids = batch_class_ids[1:]
        # Target word IDs are not used by the network. We need them to compute
        # probabilities for out-of-shortlist word.
        target_word_ids = batch_word_ids[1:]

        logprobs = tensor.log(network.target_probs())
        # Add logprobs from the class membership of the predicted word.
        logprobs += tensor.log(membership_probs)

        mask = network.mask
        if use_shortlist and network.oos_logprobs is not None:
            # The probability of out-of-shortlist words (which is the <unk>
            # probability) is multiplied by the fraction of the actual word
            # within the set of OOS words.
            logprobs += network.oos_logprobs[target_word_ids]
            # Always exclude OOV words when using a shortlist - No probability
            # mass is left for them.
            mask *= tensor.neq(target_word_ids, self._unk_id)
        elif exclude_unk:
            # If requested, ignore OOS and OOV probabilities.
            mask *= tensor.neq(target_word_ids, self._unk_id)
            mask *= tensor.lt(target_word_ids, shortlist_size)

        # Ignore unused input variables, because is_training is only used by
        # dropout layer.
        masked_logprobs = logprobs * tensor.cast(mask, theano.config.floatX)
        self._target_logprobs_function = theano.function(
            [batch_word_ids, batch_class_ids, membership_probs, network.mask],
            [masked_logprobs, mask],
            givens=[(network.input_word_ids, input_word_ids),
                    (network.input_class_ids, input_class_ids),
                    (network.target_class_ids, target_class_ids),
                    (network.is_training, numpy.int8(0))],
            name='target_logprobs',
            on_unused_input='ignore',
            profile=profile)

        # If some word is not in the training data, its class membership
        # probability will be zero. We want to ignore those words. Multiplying
        # by the mask is not possible, because those logprobs will be -inf.
        mask *= tensor.neq(membership_probs, 0.0)
        masked_logprobs = tensor.switch(mask, logprobs, 0.0)
        self._total_logprob_function = theano.function(
            [batch_word_ids, batch_class_ids, membership_probs, network.mask],
            [masked_logprobs.sum(), mask.sum()],
            givens=[(network.input_word_ids, input_word_ids),
                    (network.input_class_ids, input_class_ids),
                    (network.target_class_ids, target_class_ids),
                    (network.is_training, numpy.int8(0))],
            name='total_logprob',
            on_unused_input='ignore',
            profile=profile)

        # These are updated by score_line().
        self.num_words = 0
        self.num_unks = 0

    def score_batch(self, word_ids, class_ids, membership_probs, mask):
        """Computes the log probabilities predicted by the neural network for
        the words in a mini-batch.

        The result will be returned in a list of lists. The indices will be a
        transpose of those of the input matrices, so that the first index is the
        sequence, not the time step. The lists will contain ``None`` values in
        place of any ``<unk>`` tokens, if the constructor was given
        ``exclude_unk=True``. When using a shortlist, the lists will always
        contain ``None`` in place of OOV words, and if ``exclude_unk=True`` was
        given, also in place of OOS words. Words with zero class membership
        probability will have ``-inf`` log probability.

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
        :returns: logprob of each word in each sequence, ``None`` values
                  indicating excluded <unk> tokens
        """

        result = []
        membership_probs = membership_probs.astype(theano.config.floatX)

        # target_logprobs_function() uses the word and class IDs of the entire
        # mini-batch, but membership probs and mask are only for the output.
        logprobs, new_mask = self._target_logprobs_function(word_ids,
                                                            class_ids,
                                                            membership_probs[1:],
                                                            mask[1:])
        for seq_index in range(logprobs.shape[1]):
            seq_mask = mask[1:, seq_index]
            seq_logprobs = logprobs[seq_mask == 1, seq_index]
            # The new mask also masks excluded tokens, replace those with None.
            seq_mask = new_mask[seq_mask == 1, seq_index]
            seq_logprobs = [lp if m == 1 else None
                            for lp, m in zip(seq_logprobs, seq_mask)]
            result.append(seq_logprobs)

        return result

    def compute_perplexity(self, batch_iter):
        """Computes the perplexity of text read using the given iterator.

        ``batch_iter`` is an iterator to the input data. On each call it creates
        a two 2-dimensional matrices, both indexed by time step and sequence.
        The first matrix contains the word IDs, the second one masks out
        elements past the sequence ends.

        ``<unk>`` tokens will be excluded from the perplexity computation, if
        the constructor was given ``exclude_unk=True``. When using a shortlist,
        OOV words are always excluded, and if ``exclude_unk=True`` was given,
        OOS words are also excluded. Words with zero class membership
        probability are always excluded.

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
                self._debug_log_batch(word_ids, class_ids, membership_probs, mask)
                raise NumberError("Log probability of a mini-batch is NaN.")
            if numpy.isneginf(batch_logprob):
                self._debug_log_batch(word_ids, class_ids, membership_probs, mask)
                raise NumberError("Probability of a mini-batch is zero.")
            if batch_logprob > 0.0:
                self._debug_log_batch(word_ids, class_ids, membership_probs, mask)
                raise NumberError("Probability of a mini-batch is greater than one.")

            logprob += batch_logprob
            num_words += batch_num_words

        if num_words == 0:
            raise ValueError("Zero words for computing perplexity. Does the "
                             "evaluation data contain only OOV words?")
        cross_entropy = -logprob / num_words
        return numpy.exp(cross_entropy)

    def score_sequence(self, word_ids, class_ids, membership_probs):
        """Computes the log probability of a word sequence.

        ``<unk>`` tokens will be excluded from the probability computation, if
        the constructor was given ``exclude_unk=True``. When using a shortlist,
        OOV words are always excluded, and if ``exclude_unk=True`` was given,
        OOS words are also excluded. Words with zero class membership
        probability are always excluded.

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
            self._debug_log_batch(word_ids, class_ids, membership_probs, mask)
            raise NumberError("Log probability of a sequence is NaN.")
        if numpy.isneginf(logprob):
            self._debug_log_batch(word_ids, class_ids, membership_probs, mask)
            raise NumberError("Probability of a sequence is zero.")
        if logprob > 0.0:
            self._debug_log_batch(word_ids, class_ids, membership_probs, mask)
            raise NumberError("Probability of a sequence is greater than one.")

        return logprob

    def score_line(self, line, vocabulary):
        """Scores a line of text.

        Start-of-sentence and end-of-sentece tags (``<s>`` and ``</s>``) will be
        inserted at the beginning and the end of the line, if they're missing.
        If the line is empty, ``None`` will be returned, instead of interpreting
        it as the empty sentence ``<s> </s>``.

        ``<unk>`` tokens will be excluded from the probability computation, if
        the constructor was given ``exclude_unk=True``. When using a shortlist,
        OOV words are always excluded, and if ``exclude_unk=True`` was given,
        OOS words are also excluded. Words with zero class membership
        probability are always excluded.

        :type line: str
        :param line: a sequence of words

        :type vocabulary: Vocabulary
        :param vocabulary: vocabulary for converting the words to word IDs

        :rtype: float
        :returns: log probability of the word sequence, or None if the line is
                  empty
        """

        words = utterance_from_line(line)
        if not words:
            return None

        word_ids = vocabulary.words_to_ids(words)
        unk_id = vocabulary.word_to_id['<unk>']
        self.num_words += word_ids.size
        self.num_unks += numpy.count_nonzero(word_ids == unk_id)

        class_ids = [vocabulary.word_id_to_class_id[word_id]
                     for word_id in word_ids]
        probs = [vocabulary.get_word_prob(word_id)
                 for word_id in word_ids]

        return self.score_sequence(word_ids, class_ids, probs)

    def _debug_log_batch(self, word_ids, class_ids, membership_probs, mask):
        """Writes the target word IDs, their log probabilities, and the mask to
        the debug log.

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
        """

        membership_probs = membership_probs.astype(theano.config.floatX)

        # target_logprobs_function() uses the word and class IDs of the entire
        # mini-batch, but membership probs and mask are only for the output.
        logprobs, new_mask = self._target_logprobs_function(word_ids,
                                                            class_ids,
                                                            membership_probs[1:],
                                                            mask[1:])
        for seq_index in range(logprobs.shape[1]):
            target_word_ids = word_ids[1:, seq_index]
            seq_mask = mask[1:, seq_index]
            seq_word_ids = target_word_ids[seq_mask == 1]
            seq_logprobs = logprobs[seq_mask == 1, seq_index]
            # The new mask also masks excluded tokens.
            seq_mask = new_mask[seq_mask == 1, seq_index]
            logging.debug("Sequence %i target word IDs: [%s]",
                          seq_index, ", ".join(str(x) for x in seq_word_ids))
            logging.debug("Sequence %i mask: [%s]",
                          seq_index, ", ".join(str(x) for x in seq_mask))
            logging.debug("Sequence %i logprobs: [%s]",
                          seq_index, ", ".join(str(x) for x in seq_logprobs))
