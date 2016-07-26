#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import math
import os
import numpy
from numpy.testing import assert_equal, assert_almost_equal
import theano
from theano import tensor
from theanolm import Vocabulary
from theanolm.network import RecurrentState
from theanolm.scoring import LatticeDecoder

class DummyNetwork(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.word_input = tensor.matrix('word_input', dtype='int64')
        self.class_input = tensor.matrix('class_input', dtype='int64')
        self.target_class_ids = tensor.matrix('target_class_ids', dtype='int64')
        self.mask = tensor.matrix('mask', dtype='int64')
        self.is_training = tensor.scalar('is_training', dtype='int8')
        self.recurrent_state_input = [tensor.tensor3('recurrent_state_1', dtype=theano.config.floatX)]
        self.recurrent_state_output = [self.recurrent_state_input[0] + 1]
        self.recurrent_state_size = [3]

    def target_probs(self):
        sos_id = self.vocabulary.word_to_id['<s>']
        yksi_id = self.vocabulary.word_to_id['yksi']
        kaksi_id = self.vocabulary.word_to_id['kaksi']
        eos_id = self.vocabulary.word_to_id['</s>']

        num_time_steps = self.word_input.shape[0]
        num_sequences = self.word_input.shape[1]
        num_words = self.vocabulary.num_words()
        projection_vector = tensor.zeros(shape=(num_words,), dtype=theano.config.floatX)
        projection_vector = tensor.set_subtensor(projection_vector[sos_id], 0.1)
        projection_vector = tensor.set_subtensor(projection_vector[yksi_id], 0.2)
        projection_vector = tensor.set_subtensor(projection_vector[kaksi_id], 0.3)
        projection_vector = tensor.set_subtensor(projection_vector[eos_id], 0.4)
        result = projection_vector[self.word_input.flatten()]
        result += projection_vector[self.target_class_ids.flatten()]
        result = result.reshape([num_time_steps,
                                 num_sequences],
                                ndim=2)
        return result

class TestLatticeDecoder(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')
        with open(vocabulary_path) as vocabulary_file:
            self.vocabulary = Vocabulary.from_file(vocabulary_file, 'words')
        self.network = DummyNetwork(self.vocabulary)

        self.sos_id = self.vocabulary.word_to_id['<s>']
        self.yksi_id = self.vocabulary.word_to_id['yksi']
        self.kaksi_id = self.vocabulary.word_to_id['kaksi']
        self.eos_id = self.vocabulary.word_to_id['</s>']

    def tearDown(self):
        pass

    def test_copy_token(self):
        history = [1, 2, 3]
        token1 = LatticeDecoder.Token(history)
        token2 = LatticeDecoder.Token.copy(token1)
        token2.history.append(4)
        self.assertSequenceEqual(token1.history, [1, 2, 3])
        self.assertSequenceEqual(token2.history, [1, 2, 3, 4])

    def test_compute_total_logprob(self):
        token = LatticeDecoder.Token(ac_logprob=math.log(0.1),
                                     lat_lm_logprob=math.log(0.2),
                                     nn_lm_logprob=math.log(0.3))
        token.compute_total_logprob(0.25)
        assert_almost_equal(token.total_logprob, math.log(0.1 * (0.25 * 0.3 + 0.75 * 0.2)))
        token.compute_total_logprob(0.25, 10.0)
        assert_almost_equal(token.total_logprob, math.log(0.1) + math.log(0.25 * 0.3 + 0.75 * 0.2) * 10.0)
        token = LatticeDecoder.Token(ac_logprob=-1000,
                                     lat_lm_logprob=-1001,
                                     nn_lm_logprob=-1002)
        token.compute_total_logprob(0.75)
        # ln(exp(-1000) * (0.75 * exp(-1002) + 0.25 * exp(-1001)))
        assert_almost_equal(token.total_logprob, -2001.64263, decimal=4)

    def test_append_word(self):
        initial_state = RecurrentState(self.network.recurrent_state_size)
        token1 = LatticeDecoder.Token(history=[self.sos_id], state=initial_state)
        token2 = LatticeDecoder.Token(history=[self.sos_id, self.yksi_id], state=initial_state)
        decoder = LatticeDecoder(self.network)

        self.assertSequenceEqual(token1.history, [self.sos_id])
        self.assertSequenceEqual(token2.history, [self.sos_id, self.yksi_id])
        assert_equal(token1.state.get(0), numpy.zeros(shape=(1,1,3)).astype(theano.config.floatX))
        assert_equal(token2.state.get(0), numpy.zeros(shape=(1,1,3)).astype(theano.config.floatX))
        self.assertEqual(token1.nn_lm_logprob, 0.0)
        self.assertEqual(token2.nn_lm_logprob, 0.0)

        decoder.append_word([token1, token2], self.kaksi_id)
        self.assertSequenceEqual(token1.history, [self.sos_id, self.kaksi_id])
        self.assertSequenceEqual(token2.history, [self.sos_id, self.yksi_id, self.kaksi_id])
        assert_equal(token1.state.get(0), numpy.ones(shape=(1,1,3)).astype(theano.config.floatX))
        assert_equal(token2.state.get(0), numpy.ones(shape=(1,1,3)).astype(theano.config.floatX))
        self.assertAlmostEqual(token1.nn_lm_logprob, 0.1 + 0.3)
        self.assertAlmostEqual(token2.nn_lm_logprob, 0.2 + 0.3)

        decoder.append_word([token1, token2], self.eos_id)
        self.assertSequenceEqual(token1.history, [self.sos_id, self.kaksi_id, self.eos_id])
        self.assertSequenceEqual(token2.history, [self.sos_id, self.yksi_id, self.kaksi_id, self.eos_id])
        assert_equal(token1.state.get(0), numpy.ones(shape=(1,1,3)).astype(theano.config.floatX) * 2)
        assert_equal(token2.state.get(0), numpy.ones(shape=(1,1,3)).astype(theano.config.floatX) * 2)
        self.assertAlmostEqual(token1.nn_lm_logprob, 0.1 + 0.3 + 0.3 + 0.4)
        self.assertAlmostEqual(token2.nn_lm_logprob, 0.2 + 0.3 + 0.3 + 0.4)

if __name__ == '__main__':
    unittest.main()
