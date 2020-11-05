#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os

import numpy
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theanolm import Vocabulary, TextSampler

class DummyNetwork(object):
    def __init__(self, vocabulary):
        """
        Initialize the tensor.

        Args:
            self: (todo): write your description
            vocabulary: (todo): write your description
        """
        self.vocabulary = vocabulary

        self.input_word_ids = tensor.matrix('input_word_ids', dtype='int64')
        self.input_word_ids.tag.test_value = numpy.zeros((1, 4), dtype='int64')

        self.input_class_ids = tensor.matrix('input_class_ids', dtype='int64')
        self.input_class_ids.tag.test_value = numpy.zeros((1, 4), dtype='int64')

        self.target_class_ids = tensor.matrix('target_class_ids', dtype='int64')
        self.target_class_ids.tag.test_value = numpy.zeros((1, 4), dtype='int64')

        self.mask = tensor.matrix('mask', dtype='int64')
        self.mask.tag.test_value = numpy.zeros((1, 4), dtype='int64')

        self.is_training = tensor.scalar('is_training', dtype='int8')
        self.is_training.tag.test_value = 1

        self.recurrent_state_input = []
        self.recurrent_state_output = []
        self.recurrent_state_size = []
        self.random = RandomStreams()

    def output_probs(self):
        """
        Outputs : py : [ n_probs.

        Args:
            self: (todo): write your description
        """
        sos_id = self.vocabulary.word_to_id['<s>']
        yksi_id = self.vocabulary.word_to_id['yksi']
        kaksi_id = self.vocabulary.word_to_id['kaksi']
        eos_id = self.vocabulary.word_to_id['</s>']

        num_time_steps = self.input_word_ids.shape[0]
        num_sequences = self.input_word_ids.shape[1]
        num_words = self.vocabulary.num_shortlist_words()
        projection_matrix = tensor.zeros(shape=(num_words, num_words),
                                         dtype='float32')
        projection_matrix = tensor.set_subtensor(projection_matrix[sos_id, yksi_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[sos_id, kaksi_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[yksi_id, kaksi_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[yksi_id, eos_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[kaksi_id, yksi_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[kaksi_id, eos_id], 0.5)
        projection_matrix = tensor.set_subtensor(projection_matrix[eos_id, sos_id], 1.0)
        result = projection_matrix[self.input_word_ids.flatten()]
        result = result.reshape([num_time_steps,
                                 num_sequences,
                                 num_words],
                                ndim=3)
        return result

class TestTextSampler(unittest.TestCase):
    def setUp(self):
        """
        Set the vocab configuration.

        Args:
            self: (todo): write your description
        """
        theano.config.compute_test_value = 'warn'

        script_path = os.path.dirname(os.path.realpath(__file__))
        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')
        with open(vocabulary_path) as vocabulary_file:
            self.vocabulary = Vocabulary.from_file(vocabulary_file, 'words')
        self.dummy_network = DummyNetwork(self.vocabulary)

    def tearDown(self):
        """
        Tear down the next callable.

        Args:
            self: (todo): write your description
        """
        pass

    def test_generate(self):
        """
        Generate a sampler

        Args:
            self: (todo): write your description
        """
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
