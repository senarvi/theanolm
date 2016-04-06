#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from wordclasses.bigramoptimizer import BigramOptimizer
from wordclasses.functions import byte_size

class NumpyBigramOptimizer(BigramOptimizer):
    """Word Class Optimizer
    """

    def __init__(self, num_classes, corpus_file, vocabulary_file = None):
        """Reads the statistics from the training corpus.

        :type num_classes: int
        :param num_classes: number of classes the optimizer should create

        :type corpus_file: file object
        :param corpus_file: a file that contains the input sentences

        :type vocabulary_file: file object
        :param vocabulary_file: if not None, restricts the vocabulary to the
                                words read from this file
        """

        super().__init__(num_classes, corpus_file, vocabulary_file)

        # Read word counts from the training corpus.
        corpus_file.seek(0)
        self._word_counts, ww_counts = self._read_word_statistics(corpus_file)
        self._ww_counts = ww_counts.tocsc()
        print("Allocated {} for word counts.".format(
            byte_size(self._word_counts.nbytes)))
        print("Allocated {} for sparse word-word counts.".format(
            byte_size(self._ww_counts.data.nbytes)))

        # Initialize classes.
        self._word_to_class, self._class_to_words = \
            self._freq_init_classes(num_classes, self._word_counts)
        print("Allocated {} for word-to-class mapping.".format(
            byte_size(self._word_to_class.nbytes)))

        # Compute class counts from word counts.
        self._class_counts, self._cc_counts, self._cw_counts, self._wc_counts = \
            self._compute_class_statistics(self._word_counts,
                                           self._ww_counts,
                                           self._word_to_class)
        print("Allocated {} for class counts.".format(
            byte_size(self._class_counts.nbytes)))
        print("Allocated {} for class-class counts.".format(
            byte_size(self._cc_counts.nbytes)))
        print("Allocated {} for class-word counts.".format(
            byte_size(self._cw_counts.nbytes)))
        print("Allocated {} for word-class counts.".format(
            byte_size(self._wc_counts.nbytes)))

    def get_word_class(self, word_id):
        """Returns the class the given word is currently assigned to.

        :type word_id: int
        :param word_id: ID of the word

        :rtype: int
        :returns: ID of the class the word is assigned to
        """

        return self._word_to_class[word_id]

    def get_class_words(self, class_id):
        """Returns the words that are assigned to given class.

        :type class_id: int
        :param class_id: ID of the word

        :rtype: set
        :returns: IDs of the words that are assigned to the given class
        """

        return self._class_to_words[class_id]

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

        self._class_to_words[old_class_id].remove(word_id)
        self._class_to_words[new_class_id].add(word_id)
        self._word_to_class[word_id] = new_class_id
