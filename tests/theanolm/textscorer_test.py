#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os

import numpy
from numpy.testing import assert_almost_equal
import theano
from theano import tensor

from theanolm import Vocabulary
from theanolm.scoring import TextScorer

class DummyNetwork(object):
    """A dummy network for testing the text scorer that always outputs
    target_class_id / 100.0
    """

    def __init__(self, vocabulary):
        """
        Initialize the tensor.

        Args:
            self: (todo): write your description
            vocabulary: (todo): write your description
        """
        self.vocabulary = vocabulary
        self.input_word_ids = tensor.matrix('input_word_ids', dtype='int64')
        self.input_class_ids = tensor.matrix('input_class_ids', dtype='int64')
        self.target_class_ids = tensor.matrix('target_class_ids', dtype='int64')
        self.mask = tensor.matrix('mask', dtype='int64')
        self.oos_logprobs = theano.shared(
            numpy.log(vocabulary.get_oos_probs()),
            'network/oos_logprobs')
        self.is_training = tensor.scalar('is_training', dtype='int8')

    def target_probs(self):
        """
        The target probabilities of target probabilities.

        Args:
            self: (todo): write your description
        """
        return self.target_class_ids.astype('float32') / 100.0

class TestTextScorer(unittest.TestCase):
    def setUp(self):
        """
        Set the vocab vocabulary.

        Args:
            self: (todo): write your description
        """
        script_path = os.path.dirname(os.path.realpath(__file__))
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')
        oos_words=['yksitoista', 'kaksitoista']
        with open(vocabulary_path) as vocabulary_file:
            self.vocabulary = Vocabulary.from_file(vocabulary_file, 'words',
                                                   oos_words=oos_words)
        word_counts = {'yksi': 1, 'kaksi': 2, 'kolme': 3, 'neljä': 4,
                       'viisi': 5, 'kuusi': 6, 'seitsemän': 7, 'kahdeksan': 8,
                       'yhdeksän': 9, 'kymmenen': 10, '<s>': 11, '</s>': 12,
                       '<unk>': 13, 'yksitoista': 3, 'kaksitoista': 7}
        self.vocabulary.compute_probs(word_counts)
        self.dummy_network = DummyNetwork(self.vocabulary)

    def tearDown(self):
        """
        Tear down the next callable.

        Args:
            self: (todo): write your description
        """
        pass

    def test_score_batch(self):
        """
        Compute the model

        Args:
            self: (todo): write your description
        """
        # Network predicts <unk> probability. Out-of-shortlist words are mapped
        # to <unk> class by .
        scorer = TextScorer(self.dummy_network, use_shortlist=False)
        word_ids = numpy.arange(15).reshape((3, 5)).T
        class_ids, _ = self.vocabulary.get_class_memberships(word_ids)
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        mask = numpy.ones_like(word_ids)
        logprobs = scorer.score_batch(word_ids, class_ids, membership_probs,
                                      mask)
        assert_almost_equal(logprobs[0],
                            numpy.log(word_ids[1:,0].astype('float32') / 100.0))
        assert_almost_equal(logprobs[1],
                            numpy.log(word_ids[1:,1].astype('float32') / 100.0))
        self.assertAlmostEqual(logprobs[2][0], numpy.log(11.0 / 100.0), places=5)  # </s>
        self.assertAlmostEqual(logprobs[2][1], numpy.log(12.0 / 100.0), places=5)  # <unk>
        self.assertAlmostEqual(logprobs[2][2], numpy.log(12.0 / 100.0), places=5)
        self.assertAlmostEqual(logprobs[2][3], numpy.log(12.0 / 100.0), places=5)

        # Network predicts <unk> probability. This is distributed for
        # out-of-shortlist words according to word frequency.
        scorer = TextScorer(self.dummy_network, use_shortlist=True)
        word_ids = numpy.arange(15).reshape((3, 5)).T
        class_ids, _ = self.vocabulary.get_class_memberships(word_ids)
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        mask = numpy.ones_like(word_ids)
        logprobs = scorer.score_batch(word_ids, class_ids, membership_probs,
                                      mask)
        assert_almost_equal(logprobs[0],
                            numpy.log(word_ids[1:,0].astype('float32') / 100.0))
        assert_almost_equal(logprobs[1],
                            numpy.log(word_ids[1:,1].astype('float32') / 100.0))
        self.assertAlmostEqual(logprobs[2][0], numpy.log(11.0 / 100.0), places=5)  # </s>
        self.assertIsNone(logprobs[2][1]) # <unk>
        self.assertAlmostEqual(logprobs[2][2], numpy.log(12.0 / 100.0 * 0.3), places=5)
        self.assertAlmostEqual(logprobs[2][3], numpy.log(12.0 / 100.0 * 0.7), places=5)

        # OOV and OOS words are replaced with None.
        scorer = TextScorer(self.dummy_network, use_shortlist=False,
                            exclude_unk=True)
        word_ids = numpy.arange(15).reshape((3, 5)).T
        class_ids, _ = self.vocabulary.get_class_memberships(word_ids)
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        mask = numpy.ones_like(word_ids)
        logprobs = scorer.score_batch(word_ids, class_ids, membership_probs,
                                      mask)
        assert_almost_equal(logprobs[0],
                            numpy.log(word_ids[1:,0].astype('float32') / 100.0))
        assert_almost_equal(logprobs[1],
                            numpy.log(word_ids[1:,1].astype('float32') / 100.0))
        self.assertAlmostEqual(logprobs[2][0], numpy.log(11.0 / 100.0), places=5)  # </s>
        self.assertIsNone(logprobs[2][1]) # <unk>
        self.assertIsNone(logprobs[2][2])
        self.assertIsNone(logprobs[2][3])

    def test_score_sequence(self):
        """
        Compute the class probabilities.

        Args:
            self: (todo): write your description
        """
        # Network predicts <unk> probability.
        scorer = TextScorer(self.dummy_network, use_shortlist=False)
        word_ids = numpy.arange(15)
        class_ids, _ = self.vocabulary.get_class_memberships(word_ids)
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        logprob = scorer.score_sequence(word_ids, class_ids, membership_probs)
        correct = word_ids[1:].astype('float32')
        correct /= 100.0
        correct[12] = 12.0 / 100.0
        correct[13] = 12.0 / 100.0
        correct = numpy.log(correct).sum()
        self.assertAlmostEqual(logprob, correct, places=4)

        # Network predicts <unk> probability. This is distributed for
        # out-of-shortlist words according to word frequency.
        scorer = TextScorer(self.dummy_network, use_shortlist=True)
        word_ids = numpy.arange(15)
        class_ids, _ = self.vocabulary.get_class_memberships(word_ids)
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        logprob = scorer.score_sequence(word_ids, class_ids, membership_probs)
        correct = word_ids[1:].astype('float32')
        correct /= 100.0
        correct[11] = 1.0 # <unk> is ignored
        correct[12] = 12.0 / 100.0 * 0.3
        correct[13] = 12.0 / 100.0 * 0.7
        correct = numpy.log(correct).sum()
        self.assertAlmostEqual(logprob, correct, places=5)

        # OOV and OOS words are excluded from the resulting logprobs.
        scorer = TextScorer(self.dummy_network, use_shortlist=False,
                            exclude_unk=True)
        word_ids = numpy.arange(15)
        class_ids, _ = self.vocabulary.get_class_memberships(word_ids)
        membership_probs = numpy.ones_like(word_ids).astype('float32')
        logprob = scorer.score_sequence(word_ids, class_ids, membership_probs)
        correct = word_ids[1:12].astype('float32')
        correct /= 100.0
        correct = numpy.log(correct).sum()
        self.assertAlmostEqual(logprob, correct, places=5)

if __name__ == '__main__':
    unittest.main()
