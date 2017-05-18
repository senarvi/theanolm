#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from theanolm.vocabulary.wordclass import WordClass

class TestWordClass(unittest.TestCase):
    def setUp(self):
        pass

    def test_normalize_probs(self):
        word_class = WordClass(1, 10, 0.5)
        word_class.add(11, 1.0)
        word_class.add(12, 0.5)
        word_class.normalize_probs()
        self.assertEqual(word_class.get_prob(10), 0.25)
        self.assertEqual(word_class.get_prob(11), 0.5)
        self.assertEqual(word_class.get_prob(12), 0.25)

if __name__ == '__main__':
    unittest.main()
