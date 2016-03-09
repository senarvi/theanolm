#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import numpy
from wordclasses import Optimizer

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences_path = os.path.join(script_path, 'sentences.txt')
        self.sentences_file = open(sentences_path)

    def tearDown(self):
        self.sentences_file.close()

    def test_statistics(self):
        num_classes = 2
        num_words = 8
        optimizer = Optimizer(num_classes, self.sentences_file)
        self.assertEqual(optimizer.vocabulary_size, num_words)
        self.assertEqual(optimizer.num_classes, num_classes + 3)
        self.assertEqual(len(optimizer.word_to_class), num_words)
        self.assertEqual(len(optimizer.class_to_words), num_classes + 3)

        self.assertEqual(len(optimizer.word_counts), num_words)
        self.assertEqual(optimizer.word_counts[optimizer.word_ids['<s>']], 11)
        self.assertEqual(optimizer.word_counts[optimizer.word_ids['a']], 13)
        self.assertEqual(optimizer.word_counts[optimizer.word_ids['b']], 8)
        self.assertEqual(optimizer.word_counts[optimizer.word_ids['c']], 8)
        self.assertEqual(optimizer.word_counts[optimizer.word_ids['d']], 11)
        self.assertEqual(optimizer.word_counts[optimizer.word_ids['e']], 15)
        self.assertEqual(optimizer.word_counts[optimizer.word_ids['<UNK>']], 0)
        self.assertEqual(optimizer.word_counts[optimizer.word_ids['</s>']], 11)

        self.assertEqual(optimizer.word_word_counts.shape[0], num_words)
        self.assertEqual(optimizer.word_word_counts.shape[1], num_words)
        self.assertEqual(len(optimizer.class_counts), num_classes + 3)
        self.assertEqual(optimizer.class_class_counts.shape[0], num_classes + 3)
        self.assertEqual(optimizer.class_class_counts.shape[1], num_classes + 3)

        self.assertEqual(optimizer.class_word_counts.shape[0], num_classes + 3)
        self.assertEqual(optimizer.class_word_counts.shape[1], num_words)
        self.assertEqual(optimizer.word_class_counts.shape[0], num_words)
        self.assertEqual(optimizer.word_class_counts.shape[1], num_classes + 3)

    def test_move_and_back(self):
        num_classes = 2
        optimizer = Optimizer(num_classes, self.sentences_file)

        orig_class_counts = numpy.copy(optimizer.class_counts)
        orig_class_class_counts = numpy.copy(optimizer.class_class_counts)
        orig_class_word_counts = numpy.copy(optimizer.class_word_counts)
        orig_word_class_counts = numpy.copy(optimizer.word_class_counts)

        word_id = optimizer.word_ids['d']
        orig_class_id = optimizer.word_to_class[word_id]
        new_class_id = 3 if orig_class_id != 3 else 4
        optimizer._move(word_id, new_class_id)

        self.assertEqual(numpy.count_nonzero(optimizer.class_counts != orig_class_counts), 2)
        self.assertEqual(numpy.sum(optimizer.class_counts), numpy.sum(orig_class_counts))
        self.assertGreater(numpy.count_nonzero(optimizer.class_class_counts != orig_class_class_counts), 0)
        self.assertEqual(numpy.sum(optimizer.class_class_counts), numpy.sum(orig_class_class_counts))
        self.assertGreater(numpy.count_nonzero(optimizer.class_word_counts != orig_class_word_counts), 0)
        self.assertEqual(numpy.sum(optimizer.class_word_counts), numpy.sum(orig_class_word_counts))
        self.assertGreater(numpy.count_nonzero(optimizer.word_class_counts != orig_word_class_counts), 0)
        self.assertEqual(numpy.sum(optimizer.word_class_counts), numpy.sum(orig_word_class_counts))

        optimizer._move(word_id, orig_class_id)

        self.assertEqual(numpy.count_nonzero(optimizer.class_counts != orig_class_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer.class_class_counts != orig_class_class_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer.class_word_counts != orig_class_word_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer.word_class_counts != orig_word_class_counts), 0)

    def test_move_and_recompute(self):
        num_classes = 2

        optimizer1 = Optimizer(num_classes, self.sentences_file)
        word_id = optimizer1.word_ids['d']
        orig_class_id = optimizer1.word_to_class[word_id]
        new_class_id = 3 if orig_class_id != 3 else 4
        optimizer1.class_to_words[orig_class_id].remove(word_id)
        optimizer1.class_to_words[new_class_id].add(word_id)
        optimizer1.word_to_class[word_id] = new_class_id
        optimizer1._compute_class_statistics()

        self.sentences_file.seek(0)
        optimizer2 = Optimizer(num_classes, self.sentences_file)
        word_id = optimizer2.word_ids['d']
        orig_class_id = optimizer2.word_to_class[word_id]
        optimizer2._move(word_id, new_class_id)

        self.assertEqual(numpy.count_nonzero(optimizer1.class_counts != optimizer2.class_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer1.class_class_counts != optimizer2.class_class_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer1.class_word_counts != optimizer2.class_word_counts), 0)
        self.assertEqual(numpy.count_nonzero(optimizer1.word_class_counts != optimizer2.word_class_counts), 0)

if __name__ == '__main__':
    unittest.main()
