#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import numpy
from theano import tensor
from theanolm import Vocabulary, TextSampler
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class DummyNetwork(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.word_input = tensor.matrix('word_input', dtype='int64')
        self.class_input = tensor.matrix('class_input', dtype='int64')
        self.mask = tensor.matrix('mask', dtype='int64')
        self.is_training = tensor.scalar('network/is_training', dtype='int8')
        self.recurrent_state_input = []
        self.recurrent_state_output = []
        self.recurrent_state_size = []
        self.random = RandomStreams()

    def output_probs(self):
        sos_id = self.vocabulary.word_to_id['<s>']
        yksi_id = self.vocabulary.word_to_id['yksi']
        kaksi_id = self.vocabulary.word_to_id['kaksi']
        eos_id = self.vocabulary.word_to_id['</s>']
        sos_indices = tensor.eq(self.word_input, sos_id).nonzero()

        num_time_steps = self.word_input.shape[0]
        num_sequences = self.word_input.shape[1]
        num_words = self.vocabulary.num_words()
        projection_matrix = tensor.zeros(shape=(num_words, num_words),
                                         dtype='float32')
        projection_matrix = tensor.set_subtensor(projection_matrix[sos_id, yksi_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[sos_id, kaksi_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[yksi_id, kaksi_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[yksi_id, eos_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[kaksi_id, yksi_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[kaksi_id, eos_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[eos_id, sos_id], 1.0)
        result = projection_matrix[self.word_input.flatten()]
        result = result.reshape([num_time_steps,
                                 num_sequences,
                                 num_words],
                                ndim=3)
        return result

class TestTextSampler(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')
        with open(vocabulary_path) as vocabulary_file:
            self.vocabulary = Vocabulary.from_file(vocabulary_file, 'words')
        self.dummy_network = DummyNetwork(self.vocabulary)

    def tearDown(self):
        pass

    def test_generate(self):
        # Network predicts <unk> probability.
        sampler = TextSampler(self.dummy_network)
        words = sampler.generate(50, 10)
        self.assertEqual(len(words), 10)
        for sequence in words:
            self.assertEqual(len(sequence), 50)
            self.assertEqual(sequence[0], '<s>')
            for left, right in zip(sequence, sequence[1:]):
                if left == '<s>':
                    self.assertTrue(right == 'yksi' or right == 'kaksi')
                elif left == 'yksi':
                    self.assertTrue(right == 'kaksi' or right == '</s>')
                elif left == 'kaksi':
                    self.assertTrue(right == 'yksi' or right == '</s>')
                elif left == '</s>':
                    self.assertEqual(right, '<s>')

if __name__ == '__main__':
    unittest.main()
