#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
from wordclasses.bigramoptimizer import BigramOptimizer
from wordclasses.functions import byte_size

class NumpyBigramOptimizer(BigramOptimizer):
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

        super().__init__(vocabulary)

        # Create word counts.
        self._word_counts = statistics.unigram_counts
        logging.debug("Allocated %s for word counts.",
                      byte_size(self._word_counts.nbytes))
        self._ww_counts = statistics.bigram_counts.tocsc()
        logging.debug("Allocated %s for sparse word-word counts.",
                      byte_size(self._ww_counts.data.nbytes))

        # Initialize classes.
        self._word_to_class = numpy.array(vocabulary.word_id_to_class_id)
        logging.debug("Allocated %s for word-to-class mapping.",
                      byte_size(self._word_to_class.nbytes))

        # Compute class counts from word counts.
        logging.info("Computing class and class/word statistics.")
        self._class_counts, self._cc_counts, self._cw_counts, self._wc_counts = \
            self._compute_class_statistics(self._word_counts,
                                           self._ww_counts,
                                           self._word_to_class)

    def get_word_class(self, word_id):
        """Returns the class the given word is currently assigned to.

        :type word_id: int
        :param word_id: ID of the word

        :rtype: int
        :returns: ID of the class the word is assigned to
        """

        return self._word_to_class[word_id]

    def get_word_prob(self, word_id):
        """Returns the unigram probability of a word within its class.

        :type word_id: int
        :param word_id: ID of the word

        :rtype: float
        :returns: class membership probability
        """

        word_count = self._word_counts[word_id]
        class_id = self._word_to_class[word_id]
        class_count = self._class_counts[class_id]
        if class_count == 0:
            return 0.0
        else:
            return word_count / class_count

    def log_likelihood(self):
        """Computes the log likelihood that a bigram model would give to the
        corpus.

        :rtype: float
        :returns: log likelihood of the training corpus
        """

        return (numpy.ma.log(self._cc_counts) * self._cc_counts).sum() + \
               (numpy.ma.log(self._word_counts) * self._word_counts).sum() - \
               2 * (numpy.ma.log(self._class_counts) * self._class_counts).sum()

    def _evaluate(self, word_id, new_class_id):
        """Evaluates how much moving a word to another class would change the
        log likelihood.
        """

        old_class_id = self.get_word_class(word_id)

        # old class
        old_count = self._class_counts[old_class_id]
        new_count = old_count - self._word_counts[word_id]
        result = 2 * old_count * numpy.log(old_count)
        result -= 2 * new_count * numpy.log(new_count)

        # new class
        old_count = self._class_counts[new_class_id]
        new_count = old_count + self._word_counts[word_id]
        result += 2 * old_count * numpy.log(old_count)
        result -= 2 * new_count * numpy.log(new_count)

        # Iterate over classes other than the old and new class of the word.
        class_ids = numpy.arange(self.num_classes)
        selector = (class_ids != old_class_id) & (class_ids != new_class_id)
        iter_class_ids = class_ids[selector]

        # old class, class X
        old_counts = self._cc_counts[old_class_id,iter_class_ids]
        new_counts = old_counts - self._wc_counts[word_id,iter_class_ids]
        result -= (numpy.ma.log(old_counts) * old_counts).sum()
        result += (numpy.ma.log(new_counts) * new_counts).sum()

        # new class, class X
        old_counts = self._cc_counts[new_class_id,iter_class_ids]
        new_counts = old_counts + self._wc_counts[word_id,iter_class_ids]
        result -= (numpy.ma.log(old_counts) * old_counts).sum()
        result += (numpy.ma.log(new_counts) * new_counts).sum()

        # class X, old class
        old_counts = self._cc_counts[iter_class_ids,old_class_id]
        new_counts = old_counts - self._cw_counts[iter_class_ids,word_id]
        result -= (numpy.ma.log(old_counts) * old_counts).sum()
        result += (numpy.ma.log(new_counts) * new_counts).sum()

        # class X, new class
        old_counts = self._cc_counts[iter_class_ids,new_class_id]
        new_counts = old_counts + self._cw_counts[iter_class_ids,word_id]
        result -= (numpy.ma.log(old_counts) * old_counts).sum()
        result += (numpy.ma.log(new_counts) * new_counts).sum()

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

        return result

    def _ll_change(self, old_count, new_count):
        result = 0
        if old_count != 0:
            result -= old_count * numpy.log(old_count)
        if new_count != 0:
            result += new_count * numpy.log(new_count)
        return result

    def _move(self, word_id, new_class_id):
        """Moves a word to another class.
        """

        old_class_id = self._word_to_class[word_id]

        # word
        word_count = self._word_counts[word_id]
        self._class_counts[old_class_id] -= word_count
        self._class_counts[new_class_id] += word_count

        # word, word X
        right_word_ids = numpy.asarray(
            [id for id in self._ww_counts[word_id,:].nonzero()[1] if id != word_id])
        right_class_ids = self._word_to_class[right_word_ids]
        counts = self._ww_counts[word_id,right_word_ids].toarray().flatten()
        self._cw_counts[old_class_id,right_word_ids] -= counts
        self._cw_counts[new_class_id,right_word_ids] += counts
        numpy.add.at(self._cc_counts[old_class_id,:], right_class_ids, -counts)
        numpy.add.at(self._cc_counts[new_class_id,:], right_class_ids, counts)

        # word X, word
        left_word_ids = numpy.asarray(
            [id for id in self._ww_counts[:,word_id].nonzero()[0] if id != word_id])
        left_class_ids = self._word_to_class[left_word_ids]
        counts = self._ww_counts[left_word_ids,word_id].toarray().flatten()
        self._wc_counts[left_word_ids,old_class_id] -= counts
        self._wc_counts[left_word_ids,new_class_id] += counts
        numpy.add.at(self._cc_counts[:,old_class_id], left_class_ids, -counts)
        numpy.add.at(self._cc_counts[:,new_class_id], left_class_ids, counts)

        # word, word
        count = self._ww_counts[word_id,word_id]
        self._cc_counts[old_class_id,old_class_id] -= count
        self._cc_counts[new_class_id,new_class_id] += count
        self._cw_counts[old_class_id,word_id] -= count
        self._cw_counts[new_class_id,word_id] += count
        self._wc_counts[word_id,old_class_id] -= count
        self._wc_counts[word_id,new_class_id] += count

        self._word_to_class[word_id] = new_class_id

    def _class_size(self, class_id):
        """Calculates the number of words in a class.

        :type class_id: int
        :param class_id: ID of a class

        :rtype: int
        :returns: number of words in the class
        """

        return numpy.count_nonzero(self._word_to_class == class_id)
