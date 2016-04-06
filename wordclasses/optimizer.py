#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from scipy.sparse import csc_matrix, dok_matrix
import theano
from theano import sparse
from theano import tensor

def size(x):
    suffixes = ['bytes', 'KB', 'MB', 'GB', 'TB']
    index = 0
    while x > 1024 and index < 4:
        index += 1
        x /= 1024
    return "{} {}".format(int(round(x)), suffixes[index])

class Optimizer(object):
    """Word Class Optimizer
    """

    def __init__(self, num_classes, corpus_file, vocabulary_file = None, count_type = 'int32'):
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

        self.class_counts_theano = theano.shared(self.class_counts, 'class_counts')
        self.word_counts_theano = theano.shared(self.word_counts, 'word_counts')
        self.cc_counts_theano = theano.shared(self.cc_counts, 'cc_counts')
        self.cw_counts_theano = theano.shared(self.cw_counts, 'cw_counts')
        self.wc_counts_theano = theano.shared(self.wc_counts, 'wc_counts')
        self.ww_counts_theano = theano.shared(self.ww_counts, 'ww_counts')
        self.ww_counts_csr_theano = theano.shared(self.ww_counts_csr, 'ww_counts_csr')
        self.word_to_class_theano = theano.shared(self.word_to_class, 'word_to_class')
        self._create_evaluate_function()
        self._create_move_function()

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
            print('move_to_best_class: only one word in class {}.'.format(old_class_id))
            return False

        ll_diff, new_class_id = self._find_best_move(word_id)
        if ll_diff > 0:
            self._move_numpy(word_id, new_class_id)
            self._move_theano(word_id, new_class_id)
            return True
        else:
            return False

    def _read_word_statistics(self, corpus_file):
        """Reads word statistics from corpus file.
        """

        self.word_counts = numpy.zeros(self.vocabulary_size, self.count_type)
        print("Allocated {} for word counts.".format(
            size(self.word_counts.nbytes)))

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
        self.ww_counts_csr = self.ww_counts.tocsr()
        self.ww_counts = self.ww_counts.tocsc()
        print("Allocated {} for sparse word-word counts.".format(
            size(self.ww_counts.data.nbytes)))

    def _freq_init_classes(self, num_classes):
        """Initializes word classes based on word frequency.
        """

        self.num_classes = num_classes + 3
        self.first_normal_class_id = 3

        self.word_to_class = -1 * numpy.ones(self.vocabulary_size,
                                             self.count_type)
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
            if self.word_to_class[word_id] != -1:
                # A class has been already assigned to <s>, </s>, and <UNK>.
                continue
            self.word_to_class[word_id] = class_id
            self.class_to_words[class_id].add(word_id)
            class_id = max((class_id + 1) % self.num_classes, self.first_normal_class_id)

    def _compute_class_statistics(self):
        """Computes class statistics.
        """

        self.class_counts = numpy.zeros(self.num_classes, self.count_type)
        print("Allocated {} for class counts.".format(
            size(self.class_counts.nbytes)))

        self.cc_counts = numpy.zeros(
            (self.num_classes, self.num_classes), dtype=self.count_type)
        print("Allocated {} for class-class counts.".format(
            size(self.cc_counts.nbytes)))

        self.cw_counts = numpy.zeros(
            (self.num_classes, self.vocabulary_size), dtype=self.count_type)
        print("Allocated {} for class-word counts.".format(
            size(self.cw_counts.nbytes)))

        self.wc_counts = numpy.zeros(
            (self.vocabulary_size, self.num_classes), dtype=self.count_type)
        print("Allocated {} for word-class counts.".format(
            size(self.wc_counts.nbytes)))

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

        old_class_id_numpy = self.word_to_class[word_id]
        old_class_id_theano = self.word_to_class_theano.get_value()[word_id]
        print("old_class_id_numpy =", old_class_id_numpy, "old_class_id_theano =", old_class_id_theano)
        for class_id in range(self.first_normal_class_id, self.num_classes):
            if class_id == old_class_id_numpy:
                continue
            ll_diff_numpy = self._evaluate_numpy(word_id, class_id)
            ll_diff_theano = self._evaluate_theano(word_id, class_id)
            print("ll_diff_numpy= ", ll_diff_numpy, "ll_diff_theano =", ll_diff_theano)
            if ll_diff_numpy > best_ll_diff:
                best_ll_diff = ll_diff_numpy
                best_class_id = class_id

        assert not best_class_id is None
        return best_ll_diff, best_class_id

    def _evaluate_numpy(self, word_id, new_class_id):
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

        # Iterate over classes other than the old and new class of the word.
        class_ids = numpy.arange(self.num_classes)
        selector = (class_ids != old_class_id) & (class_ids != new_class_id)
        iter_class_ids = class_ids[selector]

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

    def _evaluate_theano(self, word_id, new_class_id):
        """Evaluates how much moving a word to another class would change the
        log likelihood.
        """

        return self._evaluate_function(word_id, new_class_id)

    def _create_evaluate_function(self):
        """Creates a Theano function that evaluates how much moving a word to
        another class would change the log likelihood.
        """

        word_id = tensor.scalar('word_id', dtype=self.count_type)
        new_class_id = tensor.scalar('new_class_id', dtype=self.count_type)
        old_class_id = self.word_to_class_theano[word_id]
        old_class_count = self.class_counts_theano[old_class_id]
        new_class_count = self.class_counts_theano[new_class_id]
        word_count = self.word_counts_theano[word_id]

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
        old_counts = self.cc_counts_theano[old_class_id,iter_class_ids]
        new_counts = old_counts - self.wc_counts_theano[word_id,iter_class_ids]
        nonzero_ids = tensor.neq(old_counts, 0)
        result -= (tensor.switch(nonzero_ids, tensor.log(old_counts), 0) * old_counts).sum()
        nonzero_ids = tensor.neq(new_counts, 0)
        result += (tensor.switch(nonzero_ids, tensor.log(new_counts), 0) * new_counts).sum()

        # new class, class X
        old_counts = self.cc_counts_theano[new_class_id,iter_class_ids]
        new_counts = old_counts + self.wc_counts_theano[word_id,iter_class_ids]
        nonzero_ids = tensor.neq(old_counts, 0)
        result -= (tensor.switch(nonzero_ids, tensor.log(old_counts), 0) * old_counts).sum()
        nonzero_ids = tensor.neq(new_counts, 0)
        result += (tensor.switch(nonzero_ids, tensor.log(new_counts), 0) * new_counts).sum()

        # class X, old class
        old_counts = self.cc_counts_theano[iter_class_ids,old_class_id]
        new_counts = old_counts - self.cw_counts_theano[iter_class_ids,word_id]
        nonzero_ids = tensor.neq(old_counts, 0)
        result -= (tensor.switch(nonzero_ids, tensor.log(old_counts), 0) * old_counts).sum()
        nonzero_ids = tensor.neq(new_counts, 0)
        result += (tensor.switch(nonzero_ids, tensor.log(new_counts), 0) * new_counts).sum()

        # class X, new class
        old_counts = self.cc_counts_theano[iter_class_ids,new_class_id]
        new_counts = old_counts + self.cw_counts_theano[iter_class_ids,word_id]
        nonzero_ids = tensor.neq(old_counts, 0)
        result -= (tensor.switch(nonzero_ids, tensor.log(old_counts), 0) * old_counts).sum()
        nonzero_ids = tensor.neq(new_counts, 0)
        result += (tensor.switch(nonzero_ids, tensor.log(new_counts), 0) * new_counts).sum()

        # old class, new class
        old_count = self.cc_counts_theano[old_class_id,new_class_id]
        new_count = old_count - \
                    self.wc_counts_theano[word_id,new_class_id] + \
                    self.cw_counts_theano[old_class_id,word_id] - \
                    self.ww_counts_theano[word_id,word_id]
        result += self._ll_change_theano(old_count, new_count)

        # new class, old class
        old_count = self.cc_counts_theano[new_class_id,old_class_id]
        new_count = old_count - \
                    self.cw_counts_theano[new_class_id,word_id] + \
                    self.wc_counts_theano[word_id,old_class_id] - \
                    self.ww_counts_theano[word_id,word_id]
        result += self._ll_change_theano(old_count, new_count)

        # old class, old class
        old_count = self.cc_counts_theano[old_class_id,old_class_id]
        new_count = old_count - \
                    self.wc_counts_theano[word_id,old_class_id] - \
                    self.cw_counts_theano[old_class_id,word_id] + \
                    self.ww_counts_theano[word_id,word_id]
        result += self._ll_change_theano(old_count, new_count)

        # new class, new class
        old_count = self.cc_counts_theano[new_class_id,new_class_id]
        new_count = old_count + \
                    self.wc_counts_theano[word_id,new_class_id] + \
                    self.cw_counts_theano[new_class_id,word_id] + \
                    self.ww_counts_theano[word_id,word_id]
        result += self._ll_change_theano(old_count, new_count)

        self._evaluate_function = theano.function(
            [word_id, new_class_id],
            result,
            name='evaluate')

    def _ll_change(self, old_count, new_count):
        result = 0
        if old_count != 0:
            result -= old_count * numpy.log(old_count)
        if new_count != 0:
            result += new_count * numpy.log(new_count)
        return result

    def _ll_change_theano(self, old_count, new_count):
        result = 0
        if old_count != 0:
            result -= old_count * tensor.log(old_count)
        if new_count != 0:
            result += new_count * tensor.log(new_count)
        return result

    def _move_numpy(self, word_id, new_class_id):
        """Moves a word to another class and updates NumPy arrays.
        """

        old_class_id = self.word_to_class[word_id]

        # word
        word_count = self.word_counts[word_id]
        self.class_counts[old_class_id] -= word_count
        self.class_counts[new_class_id] += word_count

        # word, word X
        right_word_ids = numpy.asarray(
            [id for id in self.ww_counts[word_id,:].nonzero()[1] if id != word_id])
        right_class_ids = self.word_to_class[right_word_ids]
        counts = self.ww_counts[word_id,right_word_ids].toarray().flatten()
        self.cw_counts[old_class_id,right_word_ids] -= counts
        self.cw_counts[new_class_id,right_word_ids] += counts
        numpy.add.at(self.cc_counts[old_class_id,:], right_class_ids, -counts)
        numpy.add.at(self.cc_counts[new_class_id,:], right_class_ids, counts)

        # word X, word
        left_word_ids = numpy.asarray(
            [id for id in self.ww_counts[:,word_id].nonzero()[0] if id != word_id])
        left_class_ids = self.word_to_class[left_word_ids]
        counts = self.ww_counts[left_word_ids,word_id].toarray().flatten()
        self.wc_counts[left_word_ids,old_class_id] -= counts
        self.wc_counts[left_word_ids,new_class_id] += counts
        numpy.add.at(self.cc_counts[:,old_class_id], left_class_ids, -counts)
        numpy.add.at(self.cc_counts[:,new_class_id], left_class_ids, counts)

        # word, word
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

    def _move_theano(self, word_id, new_class_id):
        """Moves a word to another class and updates Theano tensors.
        """

        self._move_function(word_id, new_class_id)
#        self.class_to_words[old_class_id].remove(word_id)
#        self.class_to_words[new_class_id].add(word_id)

    def _create_move_function(self):
        """Creates a Theano function that moves a word to another class.

        tensor.inc_subtensor actually works like numpy.add.at, so we can use it
        add the count as many times as the word occurs in a class.
        """

        updates = []
        word_id = tensor.scalar('word_id', dtype=self.count_type)
        new_class_id = tensor.scalar('new_class_id', dtype=self.count_type)
        old_class_id = self.word_to_class_theano[word_id]

        # word
        word_count = self.word_counts_theano[word_id]
        c_counts = self.class_counts_theano
        c_counts = tensor.inc_subtensor(c_counts[old_class_id], -word_count)
        c_counts = tensor.inc_subtensor(c_counts[new_class_id], word_count)
        updates.append((self.class_counts_theano, c_counts))

        # word, word X
        data, indices, indptr, _ = sparse.csm_properties(self.ww_counts_csr_theano)
        right_word_ids = indices[indptr[word_id]:indptr[word_id + 1]]
        counts = data[indptr[word_id]:indptr[word_id + 1]]
        selector = tensor.neq(right_word_ids, word_id).nonzero()
        right_word_ids = right_word_ids[selector]
        counts = counts[selector]

        cw_counts = self.cw_counts_theano
        cw_counts = tensor.inc_subtensor(cw_counts[old_class_id,right_word_ids], -counts)
        cw_counts = tensor.inc_subtensor(cw_counts[new_class_id,right_word_ids], counts)
        right_class_ids = self.word_to_class_theano[right_word_ids]
        cc_counts = self.cc_counts_theano
        cc_counts = tensor.inc_subtensor(cc_counts[old_class_id,right_class_ids], -counts)
        cc_counts = tensor.inc_subtensor(cc_counts[new_class_id,right_class_ids], counts)

        # word X, word
        data, indices, indptr, _ = sparse.csm_properties(self.ww_counts_theano)
        left_word_ids = indices[indptr[word_id]:indptr[word_id + 1]]
        counts = data[indptr[word_id]:indptr[word_id + 1]]
        selector = tensor.neq(left_word_ids, word_id).nonzero()
        left_word_ids = left_word_ids[selector]
        counts = counts[selector]

        wc_counts = self.wc_counts_theano
        wc_counts = tensor.inc_subtensor(wc_counts[left_word_ids,old_class_id], -counts)
        wc_counts = tensor.inc_subtensor(wc_counts[left_word_ids,new_class_id], counts)
        left_class_ids = self.word_to_class_theano[left_word_ids]
        cc_counts = tensor.inc_subtensor(cc_counts[left_class_ids,old_class_id], -counts)
        cc_counts = tensor.inc_subtensor(cc_counts[left_class_ids,new_class_id], counts)

        # word, word
        count = self.ww_counts_theano[word_id,word_id]
        cc_counts = tensor.inc_subtensor(cc_counts[old_class_id,old_class_id], -count)
        cc_counts = tensor.inc_subtensor(cc_counts[new_class_id,new_class_id], count)
        cw_counts = tensor.inc_subtensor(cw_counts[old_class_id,word_id], -count)
        cw_counts = tensor.inc_subtensor(cw_counts[new_class_id,word_id], count)
        wc_counts = tensor.inc_subtensor(wc_counts[word_id,old_class_id], -count)
        wc_counts = tensor.inc_subtensor(wc_counts[word_id,new_class_id], count)
        updates.append((self.cc_counts_theano, cc_counts))
        updates.append((self.cw_counts_theano, cw_counts))
        updates.append((self.wc_counts_theano, wc_counts))

        w_to_c = self.word_to_class_theano
        w_to_c = tensor.set_subtensor(w_to_c[word_id], new_class_id)
        updates.append((self.word_to_class_theano, w_to_c))

        self._move_function = theano.function(
            [word_id, new_class_id],
            [],
            updates=updates,
            name='move')
