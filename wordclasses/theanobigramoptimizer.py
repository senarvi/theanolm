#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
import theano
from theano import sparse
from theano import tensor
from wordclasses.bigramoptimizer import BigramOptimizer
from wordclasses.functions import byte_size

class TheanoBigramOptimizer(BigramOptimizer):
    """Word Class Optimizer
    """

    def __init__(self, statistics, vocabulary):
        """Computes initial statistics.

        :type statistics: WordStatistics
        :param statistics: word statistics from the training corpus

        :type vocabulary: theanolm.Vocabulary
        :param vocabulary: words to include in the optimization and initial classes
        """

	# count_nonzero() and any() seem to fail on the sparse matrix.
        if not statistics.unigram_counts.any():
            raise ValueError("Empty word unigram statistics.")
        if statistics.bigram_counts.nnz == 0:
            raise ValueError("Empty word bigram statistics.")

        # Sparse classes in Theano 0.8 support only int32 indices.
        super().__init__(vocabulary, 'int32')

        # Create word counts.
        word_counts = statistics.unigram_counts
        self._word_counts = theano.shared(word_counts, 'word_counts')
        logging.debug("Allocated %s for word counts.",
                      byte_size(word_counts.nbytes))
        ww_counts_csc = statistics.bigram_counts.tocsc()
        self._ww_counts = theano.shared(ww_counts_csc, 'ww_counts_csc')
        logging.debug("Allocated %s for CSC word-word counts.",
                      byte_size(ww_counts_csc.data.nbytes))
        ww_counts_csr = statistics.bigram_counts.tocsr()
        self._ww_counts_csr = theano.shared(ww_counts_csr, 'ww_counts_csr')
        logging.debug("Allocated %s for CSR word-word counts.",
                      byte_size(ww_counts_csr.data.nbytes))

        # Initialize classes.
        word_to_class = numpy.array(vocabulary.word_id_to_class_id)
        self._word_to_class = theano.shared(word_to_class, 'word_to_class')
        logging.debug("Allocated %s for word-to-class mapping.",
                      byte_size(word_to_class.nbytes))

        # Compute class counts from word counts.
        logging.info("Computing class and class/word statistics.")
        class_counts, cc_counts, cw_counts, wc_counts = \
            self._compute_class_statistics(word_counts,
                                           ww_counts_csc,
                                           word_to_class)
        self._class_counts = theano.shared(class_counts, 'class_counts')
        self._cc_counts = theano.shared(cc_counts, 'cc_counts')
        self._cw_counts = theano.shared(cw_counts, 'cw_counts')
        self._wc_counts = theano.shared(wc_counts, 'wc_counts')

        # Create Theano functions.
        self._create_get_word_prob_function()
        self._create_evaluate_function()
        self._create_move_function()
        self._create_log_likelihood_function()
        self._create_class_size_function()

    def get_word_class(self, word_id):
        """Returns the class the given word is currently assigned to.

        :type word_id: int
        :param word_id: ID of the word

        :rtype: int
        :returns: ID of the class the word is assigned to
        """

        return self._word_to_class.get_value()[word_id]

    def _create_get_word_prob_function(self):
        """Creates a Theano function that returns the unigram probability of a
        word within its class.
        """

        word_id = tensor.scalar('word_id', dtype=self._count_type)

        word_count = self._word_counts[word_id]
        class_id = self._word_to_class[word_id]
        class_count = self._class_counts[class_id]
        result = tensor.switch(tensor.neq(class_count, 0),
                               word_count / class_count,
                               0)

        self.get_word_prob = theano.function(
            [word_id],
            result,
            name='get_word_prob')

    def _create_evaluate_function(self):
        """Creates a Theano function that evaluates how much moving a word to
        another class would change the log likelihood.
        """

        word_id = tensor.scalar('word_id', dtype=self._count_type)
        new_class_id = tensor.scalar('new_class_id', dtype=self._count_type)
        old_class_id = self._word_to_class[word_id]
        old_class_count = self._class_counts[old_class_id]
        new_class_count = self._class_counts[new_class_id]
        word_count = self._word_counts[word_id]

        # old class
        old_count = old_class_count
        new_count = old_count - word_count
        result = 2 * old_count * tensor.log(old_count)
        result -= 2 * new_count * tensor.log(new_count)

        # new class
        old_count = new_class_count
        new_count = old_count + word_count
        result += 2 * old_count * tensor.log(old_count)
        result -= 2 * new_count * tensor.log(new_count)

        # Iterate over classes other than the old and new class of the word.
        class_ids = tensor.arange(self.num_classes)
        selector = tensor.neq(class_ids, old_class_id) * \
                   tensor.neq(class_ids, new_class_id)
        iter_class_ids = class_ids[selector.nonzero()]

        # old class, class X
        old_counts = self._cc_counts[old_class_id,iter_class_ids]
        new_counts = old_counts - self._wc_counts[word_id,iter_class_ids]
        result -= self._xlogx(old_counts).sum()
        result += self._xlogx(new_counts).sum()

        # new class, class X
        old_counts = self._cc_counts[new_class_id,iter_class_ids]
        new_counts = old_counts + self._wc_counts[word_id,iter_class_ids]
        result -= self._xlogx(old_counts).sum()
        result += self._xlogx(new_counts).sum()

        # class X, old class
        old_counts = self._cc_counts[iter_class_ids,old_class_id]
        new_counts = old_counts - self._cw_counts[iter_class_ids,word_id]
        result -= self._xlogx(old_counts).sum()
        result += self._xlogx(new_counts).sum()

        # class X, new class
        old_counts = self._cc_counts[iter_class_ids,new_class_id]
        new_counts = old_counts + self._cw_counts[iter_class_ids,word_id]
        result -= self._xlogx(old_counts).sum()
        result += self._xlogx(new_counts).sum()

        # old class, new class
        old_count = self._cc_counts[old_class_id,new_class_id]
        new_count = old_count - \
                    self._wc_counts[word_id,new_class_id] + \
                    self._cw_counts[old_class_id,word_id] - \
                    self._ww_counts[word_id,word_id]
        result += self._ll_change(old_count, new_count)

        # new class, old class
        old_count = self._cc_counts[new_class_id,old_class_id]
        new_count = old_count - \
                    self._cw_counts[new_class_id,word_id] + \
                    self._wc_counts[word_id,old_class_id] - \
                    self._ww_counts[word_id,word_id]
        result += self._ll_change(old_count, new_count)

        # old class, old class
        old_count = self._cc_counts[old_class_id,old_class_id]
        new_count = old_count - \
                    self._wc_counts[word_id,old_class_id] - \
                    self._cw_counts[old_class_id,word_id] + \
                    self._ww_counts[word_id,word_id]
        result += self._ll_change(old_count, new_count)

        # new class, new class
        old_count = self._cc_counts[new_class_id,new_class_id]
        new_count = old_count + \
                    self._wc_counts[word_id,new_class_id] + \
                    self._cw_counts[new_class_id,word_id] + \
                    self._ww_counts[word_id,word_id]
        result += self._ll_change(old_count, new_count)

        self._evaluate = theano.function(
            [word_id, new_class_id],
            result,
            name='evaluate')

    def _create_move_function(self):
        """Creates a Theano function that moves a word to another class.

        tensor.inc_subtensor actually works like numpy.add.at, so we can use it
        to add the count as many times as the word occurs in a class.
        """

        updates = []
        word_id = tensor.scalar('word_id', dtype=self._count_type)
        new_class_id = tensor.scalar('new_class_id', dtype=self._count_type)
        old_class_id = self._word_to_class[word_id]

        # word
        word_count = self._word_counts[word_id]
        c_counts = self._class_counts
        c_counts = tensor.inc_subtensor(c_counts[old_class_id], -word_count)
        c_counts = tensor.inc_subtensor(c_counts[new_class_id], word_count)
        updates.append((self._class_counts, c_counts))

        # word, word X
        data, indices, indptr, _ = sparse.csm_properties(self._ww_counts_csr)
        right_word_ids = indices[indptr[word_id]:indptr[word_id + 1]]
        counts = data[indptr[word_id]:indptr[word_id + 1]]
        selector = tensor.neq(right_word_ids, word_id).nonzero()
        right_word_ids = right_word_ids[selector]
        counts = counts[selector]

        cw_counts = self._cw_counts
        cw_counts = tensor.inc_subtensor(cw_counts[old_class_id,right_word_ids], -counts)
        cw_counts = tensor.inc_subtensor(cw_counts[new_class_id,right_word_ids], counts)
        right_class_ids = self._word_to_class[right_word_ids]
        cc_counts = self._cc_counts
        cc_counts = tensor.inc_subtensor(cc_counts[old_class_id,right_class_ids], -counts)
        cc_counts = tensor.inc_subtensor(cc_counts[new_class_id,right_class_ids], counts)

        # word X, word
        data, indices, indptr, _ = sparse.csm_properties(self._ww_counts)
        left_word_ids = indices[indptr[word_id]:indptr[word_id + 1]]
        counts = data[indptr[word_id]:indptr[word_id + 1]]
        selector = tensor.neq(left_word_ids, word_id).nonzero()
        left_word_ids = left_word_ids[selector]
        counts = counts[selector]

        wc_counts = self._wc_counts
        wc_counts = tensor.inc_subtensor(wc_counts[left_word_ids,old_class_id], -counts)
        wc_counts = tensor.inc_subtensor(wc_counts[left_word_ids,new_class_id], counts)
        left_class_ids = self._word_to_class[left_word_ids]
        cc_counts = tensor.inc_subtensor(cc_counts[left_class_ids,old_class_id], -counts)
        cc_counts = tensor.inc_subtensor(cc_counts[left_class_ids,new_class_id], counts)

        # word, word
        count = self._ww_counts[word_id,word_id]
        cc_counts = tensor.inc_subtensor(cc_counts[old_class_id,old_class_id], -count)
        cc_counts = tensor.inc_subtensor(cc_counts[new_class_id,new_class_id], count)
        cw_counts = tensor.inc_subtensor(cw_counts[old_class_id,word_id], -count)
        cw_counts = tensor.inc_subtensor(cw_counts[new_class_id,word_id], count)
        wc_counts = tensor.inc_subtensor(wc_counts[word_id,old_class_id], -count)
        wc_counts = tensor.inc_subtensor(wc_counts[word_id,new_class_id], count)
        updates.append((self._cc_counts, cc_counts))
        updates.append((self._cw_counts, cw_counts))
        updates.append((self._wc_counts, wc_counts))

        w_to_c = self._word_to_class
        w_to_c = tensor.set_subtensor(w_to_c[word_id], new_class_id)
        updates.append((self._word_to_class, w_to_c))

        self._move = theano.function(
            [word_id, new_class_id],
            [],
            updates=updates,
            name='move')

    def _create_log_likelihood_function(self):
        """Creates a Theano function that computes the log likelihood that a
        bigram model would give to the corpus.
        """

        result = self._xlogx(self._cc_counts).sum() + \
                 self._xlogx(self._word_counts).sum() - \
                 2 * self._xlogx(self._class_counts).sum()

        self.log_likelihood = theano.function(
            [],
            result,
            name='log_likelihood')

    def _create_class_size_function(self):
        """Creates a function that calculates the number of words in a class.

        :type class_id: int
        :param class_id: ID of a class

        :rtype: int
        :returns: number of words in the class
        """

        class_id = tensor.scalar('class_id', dtype=self._count_type)

        result = tensor.eq(self._word_to_class, class_id).sum()

        self._class_size = theano.function(
            [class_id],
            result,
            name='class_size')

    @staticmethod
    def _ll_change(old_count, new_count):
        """A helper function that computes the log likelihood change when a
        count changes.
        """

        result = 0
        if old_count != 0:
            result -= old_count * tensor.log(old_count)
        if new_count != 0:
            result += new_count * tensor.log(new_count)
        return result

    @staticmethod
    def _xlogx(x):
        """A helper function that computes ``x * log(x)``, where ``x`` is a
        vector that may contain zeros.
        """

        return tensor.switch(tensor.neq(x, 0), tensor.log(x), 0) * x
