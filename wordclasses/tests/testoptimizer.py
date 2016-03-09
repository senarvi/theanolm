#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
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
        self.assertEqual(optimizer.word_bigram_counts.shape[0], num_words)
        self.assertEqual(optimizer.word_bigram_counts.shape[1], num_words)
        self.assertEqual(len(optimizer.class_counts), num_classes + 3)
        self.assertEqual(optimizer.class_bigram_counts.shape[0], num_classes + 3)
        self.assertEqual(optimizer.class_bigram_counts.shape[1], num_classes + 3)

if __name__ == '__main__':
    unittest.main()
