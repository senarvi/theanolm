#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os

from theanolm.vocabulary import compute_word_counts, BigramStatistics
from theanolm import Vocabulary

class TestStatistics(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences_path = os.path.join(script_path, 'sentences4.txt')
        self.sentences_file = open(sentences_path)

    def tearDown(self):
        self.sentences_file.close()

    def test_compute_word_counts(self):
        self.sentences_file.seek(0)
        word_counts = compute_word_counts([self.sentences_file])
        self.assertEqual(word_counts['a'], 13)
        self.assertEqual(word_counts['b'], 8)
        self.assertEqual(word_counts['c'], 8)
        self.assertEqual(word_counts['d'], 11)
        self.assertEqual(word_counts['e'], 15)
        self.assertEqual(word_counts['<s>'], 11)
        self.assertEqual(word_counts['</s>'], 11)

    def test_bigram_statistics(self):
        self.sentences_file.seek(0)
        word_counts = compute_word_counts([self.sentences_file])
        self.vocabulary = Vocabulary.from_word_counts(word_counts)
        self.sentences_file.seek(0)
        statistics = BigramStatistics([self.sentences_file], self.vocabulary)

        unigram_counts = statistics.unigram_counts
        vocabulary = self.vocabulary
        self.assertEqual(unigram_counts[vocabulary.word_to_id['a']], 13)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['b']], 8)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['c']], 8)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['d']], 11)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['e']], 15)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['<unk>']], 0)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['<s>']], 11)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['</s>']], 11)

        bigram_counts = statistics.bigram_counts
        vocabulary = self.vocabulary
        a_id = vocabulary.word_to_id['a']
        b_id = vocabulary.word_to_id['b']
        self.assertEqual(bigram_counts[a_id,a_id], 3)
        self.assertEqual(bigram_counts[a_id,b_id], 2)
        self.assertEqual(bigram_counts[b_id,a_id], 1)
        self.assertEqual(bigram_counts[b_id,b_id], 0)

if __name__ == '__main__':
    unittest.main()
