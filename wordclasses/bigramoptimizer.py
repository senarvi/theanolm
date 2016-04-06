#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy
from scipy.sparse import dok_matrix

class BigramOptimizer(object):
    """Word Class Optimizer
    """

    def __init__(self, num_classes, corpus_file, vocabulary_file = None, count_type = 'int64'):
        """Reads the vocabulary from the text ``corpus_file`` and creates word
        IDs. The vocabulary may be restricted by ``vocabulary_file``.

        :type num_classes: int
        :param num_classes: number of classes the optimizer should create

        :type corpus_file: file object
        :param corpus_file: a file that contains the input sentences

        :type vocabulary_file: file object
        :param vocabulary_file: if not None, restricts the vocabulary to the
                                words read from this file
        """

        self._count_type = count_type

        self.vocabulary = set(['<s>', '</s>', '<unk>'])
        if vocabulary_file is None:
            for line in corpus_file:
                self.vocabulary.update(line.split())
        else:
            restrict_words = set(line.strip() for line in vocabulary_file)
            for line in corpus_file:
                for word in line.split():
                    if word in restrict_words:
                        self.vocabulary.add(word)

        self.vocabulary_size = len(self.vocabulary)
        self._word_ids = dict(zip(self.vocabulary, range(self.vocabulary_size)))

    def _read_word_statistics(self, corpus_file):
        """Reads word statistics from corpus file.

        :type corpus_file: file object
        :param corpus_file: a file that contains the input sentences
        """

        word_counts = numpy.zeros(self.vocabulary_size, self._count_type)
        ww_counts = dok_matrix(
            (self.vocabulary_size, self.vocabulary_size), dtype=self._count_type)

        for line in corpus_file:
            sentence = [self.get_word_id('<s>')]
            for word in line.split():
                if word in self.vocabulary:
                    sentence.append(self.get_word_id(word))
                else:
                    sentence.append(self.get_word_id('<unk>'))
            sentence.append(self.get_word_id('</s>'))
            for word_id in sentence:
                word_counts[word_id] += 1
            for left_word_id, right_word_id in zip(sentence[:-1], sentence[1:]):
                ww_counts[left_word_id,right_word_id] += 1

        return word_counts, ww_counts

    def _compute_class_statistics(self, word_counts, ww_counts, word_to_class):
        """Computes class statistics from word statistics.
        """

        class_counts = numpy.zeros(self.num_classes, self._count_type)
        cc_counts = numpy.zeros(
            (self.num_classes, self.num_classes), dtype=self._count_type)
        cw_counts = numpy.zeros(
            (self.num_classes, self.vocabulary_size), dtype=self._count_type)
        wc_counts = numpy.zeros(
            (self.vocabulary_size, self.num_classes), dtype=self._count_type)

        for word_id, class_id in enumerate(word_to_class):
            class_counts[class_id] += word_counts[word_id]
        for left_word_id, right_word_id in zip(*ww_counts.nonzero()):
            count = ww_counts[left_word_id, right_word_id]
            left_class_id = word_to_class[left_word_id]
            right_class_id = word_to_class[right_word_id]
            cc_counts[left_class_id,right_class_id] += count
            cw_counts[left_class_id,right_word_id] += count
            wc_counts[left_word_id,right_class_id] += count

        return class_counts, cc_counts, cw_counts, wc_counts

    def _freq_init_classes(self, num_classes, word_counts):
        """Initializes word classes based on word frequency.

        :type num_classes: int
        :param num_classes: number of classes to create in addition to the
                            special classes

        :type word_counts: dict
        :param word_counts: word unigram counts in the training corpus

        :rtype: (numpy.ndarray, list of sets)
        :returns: a tuple containing word-to-class and class-to-words mappings
        """

        self.num_classes = num_classes + 3
        self.first_normal_class_id = 3

        word_to_class = -1 * numpy.ones(self.vocabulary_size,
                                        self._count_type)
        class_to_words = [set() for _ in range(self.num_classes)]

        word_to_class[self.get_word_id('<s>')] = 0
        class_to_words[0].add(self.get_word_id('<s>'))
        word_to_class[self.get_word_id('</s>')] = 1
        class_to_words[1].add(self.get_word_id('</s>'))
        word_to_class[self.get_word_id('<unk>')] = 2
        class_to_words[2].add(self.get_word_id('<unk>'))

        class_id = self.first_normal_class_id
        for word_id, _ in sorted(enumerate(word_counts),
                                 key=lambda x: x[1]):
            if word_to_class[word_id] != -1:
                # A class has been already assigned to <s>, </s>, and <unk>.
                continue
            word_to_class[word_id] = class_id
            class_to_words[class_id].add(word_id)
            class_id = max((class_id + 1) % self.num_classes, self.first_normal_class_id)

        return word_to_class, class_to_words

    def move_to_best_class(self, word):
        """Moves a word to the class that minimizes training set log likelihood.
        """

        if word.startswith('<') and word.endswith('>'):
            return False

        word_id = self.get_word_id(word)
        old_class_id = self.get_word_class(word_id)
        if len(self.get_class_words(old_class_id)) == 1:
            print('move_to_best_class: only one word in class {}.'.format(old_class_id))
            return False

        ll_diff, new_class_id = self._find_best_move(word_id)
        if ll_diff > 0:
            self._move(word_id, new_class_id)
            return True
        else:
            return False

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
        for class_id in range(self.first_normal_class_id, self.num_classes):
            if class_id == old_class_id:
                continue
            ll_diff = self._evaluate(word_id, class_id)
            if ll_diff > best_ll_diff:
                best_ll_diff = ll_diff
                best_class_id = class_id

        return best_ll_diff, best_class_id

    @abc.abstractmethod
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

    @abc.abstractmethod
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

        return self._word_ids[word]

    def words(self):
        """A generator for iterating through the words.

        :rtype: (str, int, float)
        :returns: a tuple containing a word, its class ID, and unigram class
                  membership probability
        """

        for word, word_id in self._word_ids.items():
            class_id = self.get_word_class(word_id)
            prob = 0.0
            yield word, class_id, prob

    @abc.abstractmethod
    def get_word_class(self, word_id):
        """Returns the class the given word is currently assigned to.

        :type word_id: int
        :param word_id: ID of the word

        :rtype: int
        :returns: ID of the class the word is assigned to
        """

        raise NotImplementedError("BigramOptimizer.get_word_class() has to be "
                                  "implemented by the subclass.")

    @abc.abstractmethod
    def get_class_words(self, class_id):
        """Returns the words that are assigned to given class.

        :type class_id: int
        :param class_id: ID of the word

        :rtype: set
        :returns: IDs of the words that are assigned to the given class
        """

        raise NotImplementedError("BigramOptimizer.get_class_words() has to be "
                                  "implemented by the subclass.")

    @abc.abstractmethod
    def log_likelihood(self):
        """Computes the log likelihood that a bigram model would give to the
        corpus.

        :rtype: float
        :returns: log likelihood of the training corpus
        """

        raise NotImplementedError("BigramOptimizer.log_likelihood() has to be "
                                  "implemented by the subclass.")
