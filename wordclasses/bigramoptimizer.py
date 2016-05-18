#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
from wordclasses.functions import byte_size

class BigramOptimizer(object):
    """Word Class Optimizer
    """

    def __init__(self, vocabulary, count_type = 'int32'):
        """Reads the vocabulary from the text ``corpus_file`` and creates word
        IDs. The vocabulary may be restricted by ``vocabulary_file``.

        :type corpus_file: file object
        :param corpus_file: a file that contains the input sentences

        :type vocabulary: theanolm.Vocabulary
        :param vocabulary: words to include in the optimization and initial classes
        """

        self._initial_vocabulary = vocabulary
        self.vocabulary_size = vocabulary.num_words()
        self.num_classes = vocabulary.num_classes()
        self.num_normal_classes = vocabulary.num_normal_classes
        self._count_type = count_type

    def move_to_best_class(self, word):
        """Moves a word to the class that minimizes training set log likelihood.

        :type word: int
        :param word: ID of the word to be moved
        """

        if word.startswith('<') and word.endswith('>'):
            return False

        word_id = self.get_word_id(word)
        old_class_id = self.get_word_class(word_id)
        if self._class_size(old_class_id) < 2:
            logging.debug('Less than two words in class %d. Not moving word %s.',
                          old_class_id, word)
            return False

        ll_diff, new_class_id = self._find_best_move(word_id)
        if ll_diff > 0:
            self._move(word_id, new_class_id)
            return True
        else:
            return False

    def _compute_class_statistics(self, word_counts, ww_counts, word_to_class):
        """Computes class statistics from word statistics given the
        word-to-class mapping.

        :type word_counts: numpy.ndarray
        :param word_counts: word unigram counts

        :type ww_counts: scipy.sparse.csc_matrix
        :param ww_counts: word bigram counts

        :type word_to_class: numpy.ndarray
        :param word_to_class: gives the class ID of each word ID
        """

        class_counts = numpy.zeros(self.num_classes, self._count_type)
        logging.debug("Allocated %s for class counts.",
                      byte_size(class_counts.nbytes))
        cc_counts = numpy.zeros(
            (self.num_classes, self.num_classes), dtype=self._count_type)
        logging.debug("Allocated %s for class-class counts.",
                      byte_size(cc_counts.nbytes))
        cw_counts = numpy.zeros(
            (self.num_classes, self.vocabulary_size), dtype=self._count_type)
        logging.debug("Allocated %s for class-word counts.",
                      byte_size(cw_counts.nbytes))
        wc_counts = numpy.zeros(
            (self.vocabulary_size, self.num_classes), dtype=self._count_type)
        logging.debug("Allocated %s for word-class counts.",
                      byte_size(wc_counts.nbytes))

        numpy.add.at(class_counts, word_to_class, word_counts)

        left_word_ids, right_word_ids = ww_counts.nonzero()
        counts = ww_counts[left_word_ids, right_word_ids].flat
        left_class_ids = word_to_class[left_word_ids]
        right_class_ids = word_to_class[right_word_ids]
        numpy.add.at(cc_counts, (left_class_ids, right_class_ids), counts)
        numpy.add.at(cw_counts, (left_class_ids, right_word_ids), counts)
        numpy.add.at(wc_counts, (left_word_ids, right_class_ids), counts)

        return class_counts, cc_counts, cw_counts, wc_counts

    def _find_best_move(self, word_id):
        """Finds the class such that moving the given word to that class would
        give best improvement in log likelihood.

        :type word_id: int
        :param word_id: ID of the word to be moved

        :rtype: (float, int)
        :returns: a tuple containing the amount log likelihood would change and
                  the ID of the class where the word should be moved (or None if
                  there was just one word in the class)
        """

        best_ll_diff = -numpy.inf
        best_class_id = None

        old_class_id = self.get_word_class(word_id)
        for class_id in range(self.num_normal_classes):
            if class_id == old_class_id:
                continue
            ll_diff = self._evaluate(word_id, class_id)
            if ll_diff > best_ll_diff:
                best_ll_diff = ll_diff
                best_class_id = class_id

        return best_ll_diff, best_class_id

    def _evaluate(self, word_id, new_class_id):
        """Evaluates how much moving a word to another class would change the
        log likelihood.

        :type word_id: int
        :param word_id: ID of the word to be moved

        :type new_class_id: int
        :param new_class_id: ID of the class the word will be moved to

        :rtype: int
        :returns: log likelihood change
        """

        raise NotImplementedError("BigramOptimizer._evaluate() has to be "
                                  "implemented by the subclass.")

    def _move(self, word_id, new_class_id):
        """Moves a word to another class.

        :type word_id: int
        :param word_id: ID of the word to be moved

        :type new_class_id: int
        :param new_class_id: ID of the class the word will be moved to
        """

        raise NotImplementedError("BigramOptimizer._move() has to be "
                                  "implemented by the subclass.")

    def get_word_id(self, word):
        """Returns the ID of the given word.

        :type word: str
        :param word: a word

        :rtype: int
        :returns: ID of the given word
        """

        return self._initial_vocabulary.word_to_id[word]

    def words(self):
        """A generator for iterating through the words.

        :rtype: generator for (str, int, float)
        :returns: generates a tuple containing a word, its class ID, and unigram
                  class membership probability
        """

        for word, word_id in self._initial_vocabulary.word_to_id.items():
            class_id = self.get_word_class(word_id)
            prob = self.get_word_prob(word_id)
            yield word, class_id, prob

    def get_word_class(self, word_id):
        """Returns the class the given word is currently assigned to.

        :type word_id: int
        :param word_id: ID of the word

        :rtype: int
        :returns: ID of the class the word is assigned to
        """

        raise NotImplementedError("BigramOptimizer.get_word_class() has to be "
                                  "implemented by the subclass.")

    def get_word_prob(self, word_id):
        """Returns the unigram probability of a word within its class.

        :type word_id: int
        :param word_id: ID of the word

        :rtype: float
        :returns: class membership probability
        """

        raise NotImplementedError("BigramOptimizer.get_word_prob() has to be "
                                  "implemented by the subclass.")

    def log_likelihood(self):
        """Computes the log likelihood that a bigram model would give to the
        corpus.

        :rtype: float
        :returns: log likelihood of the training corpus
        """

        raise NotImplementedError("BigramOptimizer.log_likelihood() has to be "
                                  "implemented by the subclass.")

    def _class_size(self, class_id):
        """Calculates the number of words in a class.

        :type class_id: int
        :param class_id: ID of a class

        :rtype: int
        :returns: number of words in the class
        """

        raise NotImplementedError("BigramOptimizer._class_size() has to be "
                                  "implemented by the subclass.")
