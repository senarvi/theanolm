#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from scipy.sparse import csc_matrix, dok_matrix

class Optimizer(object):
    """Word Class Optimizer
    """

    def __init__(self, num_classes, corpus_file, vocabulary_file = None, count_type = numpy.int32):
        """Reads the vocabulary from the text ``corpus_file``. The vocabulary
        may be restricted by ``vocabulary_file``. Then reads the statistics from
        the text.
        """

        self.count_type = count_type

        # Read the vocabulary.
        self.vocabulary = set(['<s>', '</s>', '<UNK>'])
        if vocabulary_file is None:
            for line in corpus_file:
                self.vocabulary.update(line.split())
        else:
            restrict_words = set(line.strip() for line in vocabulary_file)
            for line in corpus_file:
                for word in line.split():
                    if word in restrict_words:
                        self.vocabulary.add(word)

        # Create word IDs and read word statistics.
        self.vocabulary_size = len(self.vocabulary)
        self.word_ids = dict(zip(self.vocabulary, range(self.vocabulary_size)))
        corpus_file.seek(0)
        self._read_word_statistics(corpus_file)

        # Initialize classes and compute class statistics.
        self._freq_init_classes(num_classes)
        self._compute_class_statistics()

    def log_likelihood(self):
        """Computes the log likelihood that a bigram model would give to the
        corpus.
        """

        return (numpy.ma.log(self.cc_counts) * self.cc_counts).sum() + \
               (numpy.ma.log(self.word_counts) * self.word_counts).sum() - \
               2 * (numpy.ma.log(self.class_counts) * self.class_counts).sum()

    def move_to_best_class(self, word):
        """Moves a word to the class that minimizes training set log likelihood.
        """

        if word.startswith('<'):
            return False

        word_id = self.word_ids[word]
        old_class_id = self.word_to_class[word_id]
        if len(self.class_to_words[old_class_id]) == 1:
            return False

        ll_diff, new_class_id = self._find_best_move(word_id)
        if ll_diff > 0:
            self._move(word_id, new_class_id)
            return True
        else:
            return False

    def _read_word_statistics(self, corpus_file):
        """Reads word statistics from corpus file.
        """

        self.word_counts = numpy.zeros(self.vocabulary_size, self.count_type)
        print("Allocated {} bytes for word counts.".format(
            self.word_counts.nbytes))

        self.ww_counts = dok_matrix(
            (self.vocabulary_size, self.vocabulary_size), dtype=self.count_type)
        for line in corpus_file:
            sentence = [self.word_ids['<s>']]
            for word in line.split():
                if word in self.vocabulary:
                    sentence.append(self.word_ids[word])
                else:
                    sentence.append(self.word_ids['<UNK>'])
            sentence.append(self.word_ids['</s>'])
            for word_id in sentence:
                self.word_counts[word_id] += 1
            for left_word_id, right_word_id in zip(sentence[:-1], sentence[1:]):
                self.ww_counts[left_word_id,right_word_id] += 1
        self.ww_counts = self.ww_counts.tocsc()
        print("Word-word counts is a sparse matrix of {} bytes.".format(
            self.ww_counts.data.nbytes))

    def _freq_init_classes(self, num_classes):
        """Initializes word classes based on word frequency.
        """

        self.num_classes = num_classes + 3
        self.first_normal_class_id = 3

        self.word_to_class = [None] * self.vocabulary_size
        self.class_to_words = [set() for _ in range(self.num_classes)]

        self.word_to_class[self.word_ids['<s>']] = 0
        self.class_to_words[0].add(self.word_ids['<s>'])
        self.word_to_class[self.word_ids['</s>']] = 1
        self.class_to_words[1].add(self.word_ids['</s>'])
        self.word_to_class[self.word_ids['<UNK>']] = 2
        self.class_to_words[2].add(self.word_ids['<UNK>'])

        class_id = self.first_normal_class_id
        for word_id, _ in sorted(enumerate(self.word_counts),
                                 key=lambda x: x[1]):
            if not self.word_to_class[word_id] is None:
                # A class has been already assigned to <s>, </s>, and <UNK>.
                continue
            self.word_to_class[word_id] = class_id
            self.class_to_words[class_id].add(word_id)
            class_id = max((class_id + 1) % self.num_classes, self.first_normal_class_id)

    def _compute_class_statistics(self):
        """Computes class statistics.
        """

        self.class_counts = numpy.zeros(self.num_classes, self.count_type)
        print("Allocated {} bytes for class counts.".format(
            self.class_counts.nbytes))

        self.cc_counts = numpy.zeros(
            (self.num_classes, self.num_classes), dtype=self.count_type)
        print("Allocated {} bytes for class-class counts.".format(
            self.cc_counts.nbytes))

        self.cw_counts = numpy.zeros(
            (self.num_classes, self.vocabulary_size), dtype=self.count_type)
        print("Allocated {} bytes for class-word counts.".format(
            self.cw_counts.nbytes))

        self.wc_counts = numpy.zeros(
            (self.vocabulary_size, self.num_classes), dtype=self.count_type)
        print("Allocated {} bytes for word-class counts.".format(
            self.wc_counts.nbytes))

        for word_id, class_id in enumerate(self.word_to_class):
            self.class_counts[class_id] += self.word_counts[word_id]
        for left_word_id, right_word_id in zip(*self.ww_counts.nonzero()):
            count = self.ww_counts[left_word_id, right_word_id]
            left_class_id = self.word_to_class[left_word_id]
            right_class_id = self.word_to_class[right_word_id]
            self.cc_counts[left_class_id,right_class_id] += count
            self.cw_counts[left_class_id,right_word_id] += count
            self.wc_counts[left_word_id,right_class_id] += count

    def _find_best_move(self, word_id):
        """Finds the class such that moving the given word to that class would
        give best improvement in log likelihood.
        """

        best_ll_diff = -numpy.inf
        best_class_id = None

        old_class_id = self.word_to_class[word_id]
        for class_id in range(self.first_normal_class_id, self.num_classes):
            if class_id == old_class_id:
                continue
            ll_diff = self._evaluate_move(word_id, class_id)
            if ll_diff > best_ll_diff:
                best_ll_diff = ll_diff
                best_class_id = class_id

        assert not best_class_id is None
        return best_ll_diff, best_class_id

    def _evaluate_move(self, word_id, new_class_id):
        """Evaluates how much moving a word to another class would change the
        log likelihood.
        """

        old_class_id = self.word_to_class[word_id]

        # old class
        old_count = self.class_counts[old_class_id]
        new_count = old_count - self.word_counts[word_id]
        result = 2 * old_count * numpy.log(old_count)
        result -= 2 * new_count * numpy.log(new_count)

        # new class
        old_count = self.class_counts[new_class_id]
        new_count = old_count + self.word_counts[word_id]
        result += 2 * old_count * numpy.log(old_count)
        result -= 2 * new_count * numpy.log(new_count)

        iter_class_ids = numpy.asarray(
            [id != old_class_id and id != new_class_id
             for id in range(self.num_classes)])

        # old class, class X
        old_counts = self.cc_counts[old_class_id,iter_class_ids]
        new_counts = old_counts - self.wc_counts[word_id,iter_class_ids]
        result -= (numpy.ma.log(old_counts) * old_counts).sum()
        result += (numpy.ma.log(new_counts) * new_counts).sum()

        # new class, class X
        old_counts = self.cc_counts[new_class_id,iter_class_ids]
        new_counts = old_counts + self.wc_counts[word_id,iter_class_ids]
        result -= (numpy.ma.log(old_counts) * old_counts).sum()
        result += (numpy.ma.log(new_counts) * new_counts).sum()

        # class X, old class
        old_counts = self.cc_counts[iter_class_ids,old_class_id]
        new_counts = old_counts - self.cw_counts[iter_class_ids,word_id]
        result -= (numpy.ma.log(old_counts) * old_counts).sum()
        result += (numpy.ma.log(new_counts) * new_counts).sum()

        # class X, new class
        old_counts = self.cc_counts[iter_class_ids,new_class_id]
        new_counts = old_counts + self.cw_counts[iter_class_ids,word_id]
        result -= (numpy.ma.log(old_counts) * old_counts).sum()
        result += (numpy.ma.log(new_counts) * new_counts).sum()

        # old class, new class
        old_count = self.cc_counts[old_class_id,new_class_id]
        new_count = old_count - \
                    self.wc_counts[word_id,new_class_id] + \
                    self.cw_counts[old_class_id,word_id] - \
                    self.ww_counts[word_id,word_id]
        result += self._ll_change(old_count, new_count)

        # new class, old class
        old_count = self.cc_counts[new_class_id,old_class_id]
        new_count = old_count - \
                    self.cw_counts[new_class_id,word_id] + \
                    self.wc_counts[word_id,old_class_id] - \
                    self.ww_counts[word_id,word_id]
        result += self._ll_change(old_count, new_count)

        # old class, old class
        old_count = self.cc_counts[old_class_id,old_class_id]
        new_count = old_count - \
                    self.wc_counts[word_id,old_class_id] - \
                    self.cw_counts[old_class_id,word_id] + \
                    self.ww_counts[word_id,word_id]
        result += self._ll_change(old_count, new_count)

        # new class, new class
        old_count = self.cc_counts[new_class_id,new_class_id]
        new_count = old_count + \
                    self.wc_counts[word_id,new_class_id] + \
                    self.cw_counts[new_class_id,word_id] + \
                    self.ww_counts[word_id,word_id]
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

        old_class_id = self.word_to_class[word_id]
        word_count = self.word_counts[word_id]
        self.class_counts[old_class_id] -= word_count
        self.class_counts[new_class_id] += word_count

        for right_word_id in self.ww_counts[word_id,:].nonzero()[1]:
            if right_word_id == word_id:
                continue
            count = self.ww_counts[word_id,right_word_id]
            right_class_id = self.word_to_class[right_word_id]
            self.cc_counts[old_class_id,right_class_id] -= count
            self.cc_counts[new_class_id,right_class_id] += count
            self.cw_counts[old_class_id,right_word_id] -= count
            self.cw_counts[new_class_id,right_word_id] += count

        for left_word_id in self.ww_counts[:,word_id].nonzero()[0]:
            if left_word_id == word_id:
                continue
            count = self.ww_counts[left_word_id,word_id]
            left_class_id = self.word_to_class[left_word_id]
            self.cc_counts[left_class_id,old_class_id] -= count
            self.cc_counts[left_class_id,new_class_id] += count
            self.wc_counts[left_word_id,old_class_id] -= count
            self.wc_counts[left_word_id,new_class_id] += count

        count = self.ww_counts[word_id,word_id]
        self.cc_counts[old_class_id,old_class_id] -= count
        self.cc_counts[new_class_id,new_class_id] += count
        self.cw_counts[old_class_id,word_id] -= count
        self.cw_counts[new_class_id,word_id] += count
        self.wc_counts[word_id,old_class_id] -= count
        self.wc_counts[word_id,new_class_id] += count

        self.class_to_words[old_class_id].remove(word_id)
        self.class_to_words[new_class_id].add(word_id)
        self.word_to_class[word_id] = new_class_id
