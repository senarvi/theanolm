#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import numpy
from wordclasses import TheanoBigramOptimizer, NumpyBigramOptimizer, WordStatistics
from theanolm import Vocabulary

class TestBigramOptimizer(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences_path = os.path.join(script_path, 'sentences.txt')
        self.sentences_file = open(sentences_path)
        self.num_classes = 2
        self.vocabulary = Vocabulary.from_corpus(self.sentences_file, self.num_classes)
        self.sentences_file.seek(0)
        self.statistics = WordStatistics(self.sentences_file, self.vocabulary)

    def tearDown(self):
        self.sentences_file.close()

    def test_statistics(self):
        num_words = 8
        theano_optimizer = TheanoBigramOptimizer(self.statistics, self.vocabulary)
        numpy_optimizer = NumpyBigramOptimizer(self.statistics, self.vocabulary)
        self.assertEqual(theano_optimizer.vocabulary_size, num_words)
        self.assertEqual(numpy_optimizer.vocabulary_size, num_words)
        self.assertEqual(theano_optimizer.num_classes, self.num_classes + 3)
        self.assertEqual(numpy_optimizer.num_classes, self.num_classes + 3)
        self.assertEqual(len(theano_optimizer._word_to_class.get_value()), num_words)
        self.assertEqual(len(numpy_optimizer._word_to_class), num_words)

        sos_word_id = self.vocabulary.word_to_id['<s>']
        a_word_id = self.vocabulary.word_to_id['a']
        b_word_id = self.vocabulary.word_to_id['b']
        c_word_id = self.vocabulary.word_to_id['c']
        d_word_id = self.vocabulary.word_to_id['d']
        e_word_id = self.vocabulary.word_to_id['e']
        unk_word_id = self.vocabulary.word_to_id['<unk>']
        eos_word_id = self.vocabulary.word_to_id['</s>']

        self.assertEqual(len(theano_optimizer._word_counts.get_value()), num_words)
        self.assertEqual(len(numpy_optimizer._word_counts), num_words)
        self.assertEqual(numpy_optimizer._word_counts[sos_word_id], 11)
        self.assertEqual(theano_optimizer._word_counts.get_value()[sos_word_id], 11)
        self.assertEqual(numpy_optimizer._word_counts[a_word_id], 13)
        self.assertEqual(theano_optimizer._word_counts.get_value()[a_word_id], 13)
        self.assertEqual(numpy_optimizer._word_counts[b_word_id], 8)
        self.assertEqual(theano_optimizer._word_counts.get_value()[b_word_id], 8)
        self.assertEqual(numpy_optimizer._word_counts[c_word_id], 8)
        self.assertEqual(theano_optimizer._word_counts.get_value()[c_word_id], 8)
        self.assertEqual(numpy_optimizer._word_counts[d_word_id], 11)
        self.assertEqual(theano_optimizer._word_counts.get_value()[d_word_id], 11)
        self.assertEqual(numpy_optimizer._word_counts[e_word_id], 15)
        self.assertEqual(theano_optimizer._word_counts.get_value()[e_word_id], 15)
        self.assertEqual(numpy_optimizer._word_counts[unk_word_id], 0)
        self.assertEqual(theano_optimizer._word_counts.get_value()[unk_word_id], 0)
        self.assertEqual(numpy_optimizer._word_counts[eos_word_id], 11)
        self.assertEqual(theano_optimizer._word_counts.get_value()[eos_word_id], 11)

        self.assertEqual(numpy_optimizer._ww_counts.shape[0], num_words)
        self.assertEqual(numpy_optimizer._ww_counts.shape[1], num_words)
        self.assertEqual(theano_optimizer._ww_counts.get_value().shape[0], num_words)
        self.assertEqual(theano_optimizer._ww_counts.get_value().shape[1], num_words)
        self.assertEqual(len(numpy_optimizer._class_counts), self.num_classes + 3)
        self.assertEqual(len(theano_optimizer._class_counts.get_value()), self.num_classes + 3)
        self.assertEqual(numpy_optimizer._cc_counts.shape[0], self.num_classes + 3)
        self.assertEqual(theano_optimizer._cc_counts.get_value().shape[1], self.num_classes + 3)

        self.assertEqual(numpy_optimizer._cw_counts.shape[0], self.num_classes + 3)
        self.assertEqual(numpy_optimizer._cw_counts.shape[1], num_words)
        self.assertEqual(theano_optimizer._cw_counts.get_value().shape[0], self.num_classes + 3)
        self.assertEqual(theano_optimizer._cw_counts.get_value().shape[1], num_words)
        self.assertEqual(numpy_optimizer._wc_counts.shape[0], num_words)
        self.assertEqual(numpy_optimizer._wc_counts.shape[1], self.num_classes + 3)
        self.assertEqual(theano_optimizer._wc_counts.get_value().shape[0], num_words)
        self.assertEqual(theano_optimizer._wc_counts.get_value().shape[1], self.num_classes + 3)

    def test_move_and_back(self):
        numpy_optimizer = NumpyBigramOptimizer(self.statistics, self.vocabulary)
        theano_optimizer = TheanoBigramOptimizer(self.statistics, self.vocabulary)

        orig_class_counts = numpy.copy(numpy_optimizer._class_counts)
        orig_cc_counts = numpy.copy(numpy_optimizer._cc_counts)
        orig_cw_counts = numpy.copy(numpy_optimizer._cw_counts)
        orig_wc_counts = numpy.copy(numpy_optimizer._wc_counts)

        word_id = self.vocabulary.word_to_id['d']
        orig_class_id = numpy_optimizer.get_word_class(word_id)
        new_class_id = 3 if orig_class_id != 3 else 4
        numpy_optimizer._move(word_id, new_class_id)

        self.assertEqual(numpy.count_nonzero(numpy_optimizer._class_counts != orig_class_counts), 2)
        self.assertEqual(numpy.sum(numpy_optimizer._class_counts), numpy.sum(orig_class_counts))
        self.assertGreater(numpy.count_nonzero(numpy_optimizer._cc_counts != orig_cc_counts), 0)
        self.assertEqual(numpy.sum(numpy_optimizer._cc_counts), numpy.sum(orig_cc_counts))
        self.assertGreater(numpy.count_nonzero(numpy_optimizer._cw_counts != orig_cw_counts), 0)
        self.assertEqual(numpy.sum(numpy_optimizer._cw_counts), numpy.sum(orig_cw_counts))
        self.assertGreater(numpy.count_nonzero(numpy_optimizer._wc_counts != orig_wc_counts), 0)
        self.assertEqual(numpy.sum(numpy_optimizer._wc_counts), numpy.sum(orig_wc_counts))

        numpy_optimizer._move(word_id, orig_class_id)

        self.assertEqual(numpy.count_nonzero(numpy_optimizer._class_counts != orig_class_counts), 0)
        self.assertEqual(numpy.count_nonzero(numpy_optimizer._cc_counts != orig_cc_counts), 0)
        self.assertEqual(numpy.count_nonzero(numpy_optimizer._cw_counts != orig_cw_counts), 0)
        self.assertEqual(numpy.count_nonzero(numpy_optimizer._wc_counts != orig_wc_counts), 0)

    def test_move_and_recompute(self):
        optimizer1 = NumpyBigramOptimizer(self.statistics, self.vocabulary)
        word_id = self.vocabulary.word_to_id['d']
        orig_class_id = optimizer1.get_word_class(word_id)
        new_class_id = 3 if orig_class_id != 3 else 4
        optimizer1._word_to_class[word_id] = new_class_id
        counts = optimizer1._compute_class_statistics(optimizer1._word_counts,
                                                      optimizer1._ww_counts,
                                                      optimizer1._word_to_class)
        optimizer1._class_counts = counts[0]
        optimizer1._cc_counts = counts[1]
        optimizer1._cw_counts = counts[2]
        optimizer1._wc_counts = counts[3]

        optimizer2 = NumpyBigramOptimizer(self.statistics, self.vocabulary)
        orig_class_id = optimizer2.get_word_class(word_id)
        optimizer2._move(word_id, new_class_id)

        self.assertEqual(numpy.count_nonzero(optimizer1._class_counts != optimizer2._class_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer1._cc_counts != optimizer2._cc_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer1._cw_counts != optimizer2._cw_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer1._wc_counts != optimizer2._wc_counts), 0)

    def test_evaluate(self):
        numpy_optimizer = NumpyBigramOptimizer(self.statistics, self.vocabulary)
        theano_optimizer = TheanoBigramOptimizer(self.statistics, self.vocabulary)
        word_id = numpy_optimizer.get_word_id('d')
        orig_class_id = numpy_optimizer.get_word_class(word_id)
        new_class_id = 3 if orig_class_id != 3 else 4

        orig_ll = numpy_optimizer.log_likelihood()
        ll_diff_numpy = numpy_optimizer._evaluate(word_id, new_class_id)
#        ll_diff_theano = theano_optimizer._evaluate(word_id, new_class_id)
#        self.assertTrue(numpy.isclose(ll_diff_numpy, ll_diff_theano))

        numpy_optimizer._move(word_id, new_class_id)
        new_ll = numpy_optimizer.log_likelihood()

        self.assertFalse(numpy.isclose(orig_ll, new_ll))
        self.assertTrue(numpy.isclose(orig_ll + ll_diff_numpy, new_ll))

if __name__ == '__main__':
    unittest.main()
