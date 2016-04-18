#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
from wordclasses import WordStatistics
from theanolm import Vocabulary

class TestStatistics(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        sentences_path = os.path.join(script_path, 'sentences.txt')
        with open(sentences_path) as sentences_file:
            self.vocabulary = Vocabulary.from_corpus(sentences_file)
            sentences_file.seek(0)
            self.statistics = WordStatistics([sentences_file], self.vocabulary)

    def test_unigram_counts(self):
        unigram_counts = self.statistics.unigram_counts
        vocabulary = self.vocabulary
        self.assertEqual(unigram_counts[vocabulary.word_to_id['<s>']], 11)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['a']], 13)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['b']], 8)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['c']], 8)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['d']], 11)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['e']], 15)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['<unk>']], 0)
        self.assertEqual(unigram_counts[vocabulary.word_to_id['</s>']], 11)

    def test_bigram_counts(self):
        bigram_counts = self.statistics.bigram_counts
        vocabulary = self.vocabulary
        a_id = vocabulary.word_to_id['a']
        b_id = vocabulary.word_to_id['b']
        self.assertEqual(bigram_counts[a_id,a_id], 3)
        self.assertEqual(bigram_counts[a_id,b_id], 2)
        self.assertEqual(bigram_counts[b_id,a_id], 1)
        self.assertEqual(bigram_counts[b_id,b_id], 0)

if __name__ == '__main__':
    unittest.main()
