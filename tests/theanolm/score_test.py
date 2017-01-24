#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from theanolm.commands.score import _merge_subwords
from numpy.testing import assert_almost_equal

class TestScore(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_merge_subwords(self):
        # word vocabulary
        subwords = ['<s>', 'aaa', 'bbb', 'ccc', 'ddd', '</s>']
        subword_logprobs = [0.0, 0.1, 0.2, 0.3, 0.4]
        words, word_logprobs = _merge_subwords(subwords, subword_logprobs, None)
        self.assertSequenceEqual(words, subwords)
        assert_almost_equal(word_logprobs, subword_logprobs)

        # subword vocabulary with word boundary token, <unk> predicted
        subwords = ['<s>', '<w>', 'aaa', '<w>', 'bbb', '<unk>', '<w>', 'ccc', 'ddd', '<w>', '</s>']
        subword_logprobs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        words, word_logprobs = _merge_subwords(subwords, subword_logprobs, "word-boundary")
        self.assertSequenceEqual(words, ['<s>', 'aaa', '<unk>', 'cccddd', '</s>'])
        assert_almost_equal(word_logprobs, [0.2 + 0.3, 0.4 + 0.5 + 0.6, 0.7 + 0.8 + 0.9, 1.0])

        # subword vocabulary with word boundary token, <unk> not predicted
        subword_logprobs = [0.1, 0.2, 0.3, 0.4, None, 0.6, 0.7, 0.8, 0.9, 1.0]
        words, word_logprobs = _merge_subwords(subwords, subword_logprobs, "word-boundary")
        self.assertSequenceEqual(words, ['<s>', 'aaa', '<unk>', 'cccddd', '</s>'])
        self.assertAlmostEqual(word_logprobs[0], 0.2 + 0.3)
        self.assertIsNone(word_logprobs[1])
        self.assertAlmostEqual(word_logprobs[2], 0.7 + 0.8 + 0.9)
        self.assertAlmostEqual(word_logprobs[3], 1.0)

        # subword vocabulary with prefix/affix markings, <unk> predicted
        subwords = ['<s>', 'aaa', 'bbb+', '+ccc', '<unk>', '</s>']
        subword_logprobs = [0.1, 0.2, 0.3, 0.4, 0.5]
        words, word_logprobs = _merge_subwords(subwords, subword_logprobs, "prefix-affix")
        self.assertSequenceEqual(words, ['<s>', 'aaa', 'bbbccc', '<unk>', '</s>'])
        assert_almost_equal(word_logprobs, [0.1, 0.2 + 0.3, 0.4, 0.5])

        # subword vocabulary with prefix/affix markings, <unk> not predicted
        subwords = ['<s>', 'aaa', 'bbb+', '+ccc', '<unk>', '</s>']
        subword_logprobs = [0.1, 0.2, 0.3, None, 0.5]
        words, word_logprobs = _merge_subwords(subwords, subword_logprobs, "prefix-affix")
        self.assertSequenceEqual(words, ['<s>', 'aaa', 'bbbccc', '<unk>', '</s>'])
        self.assertAlmostEqual(word_logprobs[0], 0.1)
        self.assertAlmostEqual(word_logprobs[1], 0.2 + 0.3)
        self.assertIsNone(word_logprobs[2])
        self.assertAlmostEqual(word_logprobs[3], 0.5)

if __name__ == '__main__':
    unittest.main()
