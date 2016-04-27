#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import numpy
from theano import tensor
from theanolm import Vocabulary, TextScorer
from numpy.testing import assert_almost_equal

class DummyNetwork(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.word_input = tensor.matrix('word_input', dtype='int64')
        self.class_input = tensor.matrix('class_input', dtype='int64')
        self.mask = tensor.matrix('mask', dtype='int64')
        self.is_training = tensor.scalar('network/is_training', dtype='int8')
        self.prediction_probs = self.word_input[1:].astype('float32') / 5

class TestTextScorer(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')
        with open(vocabulary_path) as vocabulary_file:
            self.vocabulary = Vocabulary.from_file(vocabulary_file, 'words')
        self.dummy_network = DummyNetwork(self.vocabulary)

    def tearDown(self):
        pass

    def test_score_batch(self):
        # Network predicts <unk> probability.
        scorer = TextScorer(self.dummy_network)
        word_ids = numpy.arange(6).reshape((3, 2))
        class_ids = numpy.arange(6).reshape((3, 2))
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        mask = numpy.ones_like(word_ids)
        logprobs = scorer.score_batch(word_ids, class_ids, membership_probs,
                                      mask)
        assert_almost_equal(logprobs[0],
                            numpy.log(word_ids[1:,0].astype('float32') / 5))
        assert_almost_equal(logprobs[1],
                            numpy.log(word_ids[1:,1].astype('float32') / 5))

        # <unk> is removed from the resulting logprobs.
        scorer = TextScorer(self.dummy_network, ignore_unk=True)
        word_ids = numpy.arange(6).reshape((3, 2))
        word_ids[1,1] = self.vocabulary.word_to_id['<unk>']
        class_ids = numpy.arange(6).reshape((3, 2))
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        mask = numpy.ones_like(word_ids)
        logprobs = scorer.score_batch(word_ids, class_ids, membership_probs,
                                      mask)
        assert_almost_equal(logprobs[0],
                            numpy.log(word_ids[1:,0].astype('float32') / 5))
        assert_almost_equal(logprobs[1],
                            numpy.log(word_ids[2:,1].astype('float32') / 5))

        # <unk> is assigned a constant logprob.
        scorer = TextScorer(self.dummy_network, ignore_unk=False, unk_penalty=-5)
        word_ids = numpy.arange(6).reshape((3, 2))
        word_ids[1,1] = self.vocabulary.word_to_id['<unk>']
        class_ids = numpy.arange(6).reshape((3, 2))
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        mask = numpy.ones_like(word_ids)
        logprobs = scorer.score_batch(word_ids, class_ids, membership_probs,
                                      mask)
        assert_almost_equal(logprobs[0],
                            numpy.log(word_ids[1:,0].astype('float32') / 5))
        assert_almost_equal(logprobs[1][0], -5)
        assert_almost_equal(logprobs[1][1],
                            numpy.log(word_ids[2,1].astype('float32') / 5))

    def test_score_sequence(self):
        # Network predicts <unk> probability.
        scorer = TextScorer(self.dummy_network)
        word_ids = numpy.arange(6)
        class_ids = numpy.arange(6)
        membership_probs = numpy.ones(6, dtype='float32')
        logprob = scorer.score_sequence(word_ids, class_ids, membership_probs)
        correct = word_ids[1:].astype('float32')    
        correct = correct / 5
        correct = numpy.log(correct).sum()
        self.assertAlmostEqual(logprob, correct, places=5)

        # <unk> is removed from the resulting logprobs.
        scorer = TextScorer(self.dummy_network, ignore_unk=True)
        word_ids = numpy.arange(6)
        word_ids[3] = self.vocabulary.word_to_id['<unk>']
        class_ids = numpy.arange(6)
        membership_probs = numpy.ones(6, dtype='float32')
        logprob = scorer.score_sequence(word_ids, class_ids, membership_probs)
        correct = word_ids[[1, 2, 4, 5]].astype('float32')
        correct = correct / 5
        correct = numpy.log(correct).sum()
        self.assertAlmostEqual(logprob, correct, places=5)

        # <unk> is assigned a constant logprob.
        scorer = TextScorer(self.dummy_network, ignore_unk=False, unk_penalty=-5)
        word_ids = numpy.arange(6)
        word_ids[3] = self.vocabulary.word_to_id['<unk>']
        class_ids = numpy.arange(6)
        membership_probs = numpy.ones(6, dtype='float32')
        logprob = scorer.score_sequence(word_ids, class_ids, membership_probs)
        correct = word_ids[[1, 2, 4, 5]].astype('float32')
        correct = correct / 5
        correct = numpy.log(correct).sum() - 5
        self.assertAlmostEqual(logprob, correct, places=5)

if __name__ == '__main__':
    unittest.main()
