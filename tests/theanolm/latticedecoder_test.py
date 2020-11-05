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
from theanolm.scoring.slflattice import SLFLattice
from theanolm.scoring.lattice import Lattice

class DummyNetwork(object):
    """A dummy network for testing the lattice decoder that always outputs
    projection of input word + projection of output word.
    """

    def __init__(self, vocabulary, projection_vector):
        """
        Initialize the vocab.

        Args:
            self: (todo): write your description
            vocabulary: (todo): write your description
            projection_vector: (todo): write your description
        """
        self.vocabulary = vocabulary
        self.input_word_ids = tensor.matrix('input_word_ids', dtype='int64')
        self.input_class_ids = tensor.matrix('input_class_ids', dtype='int64')
        self.target_class_ids = tensor.matrix('target_class_ids', dtype='int64')
        self.mask = tensor.matrix('mask', dtype='int64')
        self.is_training = tensor.scalar('is_training', dtype='int8')
        self.recurrent_state_input = [tensor.tensor3('recurrent_state_1', dtype=theano.config.floatX)]
        self.recurrent_state_output = [self.recurrent_state_input[0] + 1]
        self.recurrent_state_size = [3]
        self.projection_vector = projection_vector

    def target_probs(self):
        """
        Return the target probabilities.

        Args:
            self: (todo): write your description
        """
        num_time_steps = self.input_word_ids.shape[0]
        num_sequences = self.input_word_ids.shape[1]
        result = self.projection_vector[self.input_word_ids.flatten()]
        result += self.projection_vector[self.target_class_ids.flatten()]
        result = result.reshape([num_time_steps,
                                 num_sequences],
                                ndim=2)
        return result

class DummyLatticeDecoder(LatticeDecoder):
    def __init__(self):
        """
        Initialize a list of tokens.

        Args:
            self: (todo): write your description
        """
        self._sorted_nodes = [Lattice.Node(id) for id in range(5)]
        self._sorted_nodes[0].time = 0.0
        self._sorted_nodes[1].time = 1.0
        self._sorted_nodes[2].time = 1.0
        self._sorted_nodes[3].time = None
        self._sorted_nodes[4].time = 3.0
        self._tokens = [[LatticeDecoder.Token()],
                        [LatticeDecoder.Token()],
                        [LatticeDecoder.Token(), LatticeDecoder.Token(), LatticeDecoder.Token()],
                        [LatticeDecoder.Token()],
                        []]
        self._tokens[0][0].total_logprob = -10.0
        self._tokens[0][0].recombination_hash = 1
        self._sorted_nodes[0].best_logprob = -10.0
        self._tokens[1][0].total_logprob = -20.0
        self._tokens[1][0].recombination_hash = 1
        self._sorted_nodes[1].best_logprob = -20.0
        self._tokens[2][0].total_logprob = -30.0
        self._tokens[2][0].recombination_hash = 1
        self._tokens[2][1].total_logprob = -50.0
        self._tokens[2][1].recombination_hash = 2
        self._tokens[2][2].total_logprob = -70.0
        self._tokens[2][2].recombination_hash = 3
        self._sorted_nodes[2].best_logprob = -30.0
        self._tokens[3][0].total_logprob = -100.0
        self._tokens[3][0].recombination_hash = 1
        self._sorted_nodes[3].best_logprob = -100.0
        self._prune_extra_limit = None
        self._abs_min_beam = 0
        self._abs_min_max_tokens = 0

class TestLatticeDecoder(unittest.TestCase):
    def setUp(self):
        """
        Set the vocab.

        Args:
            self: (todo): write your description
        """
        self.maxDiff = None
        script_path = os.path.dirname(os.path.realpath(__file__))

        vocabulary_path = os.path.join(script_path, 'vocabulary.txt')
        with open(vocabulary_path) as vocabulary_file:
            self.vocabulary = Vocabulary.from_file(vocabulary_file, 'words',
                                                   oos_words=['oos1', 'oos2'])
        self.vocabulary.compute_probs({'yksi': 1,
                                       'kaksi': 1,
                                       'kolme': 1,
                                       'neljä': 1,
                                       'viisi': 1,
                                       'kuusi': 1,
                                       'seitsemän': 1,
                                       'kahdeksan': 1,
                                       'yhdeksän': 1,
                                       'kymmenen': 1,
                                       'oos1': 1,
                                       'oos2': 2})

        self.sos_id = self.vocabulary.word_to_id['<s>']
        self.yksi_id = self.vocabulary.word_to_id['yksi']
        self.kaksi_id = self.vocabulary.word_to_id['kaksi']
        self.eos_id = self.vocabulary.word_to_id['</s>']
        self.unk_id = self.vocabulary.word_to_id['<unk>']
        self.oos1_id = self.vocabulary.word_to_id['oos1']
        self.oos2_id = self.vocabulary.word_to_id['oos2']

        projection_vector = tensor.zeros(
            shape=(self.vocabulary.num_shortlist_words(),),
            dtype=theano.config.floatX)
        self.sos_prob = 0.1
        projection_vector = tensor.set_subtensor(
            projection_vector[self.sos_id], self.sos_prob)
        self.yksi_prob = 0.2
        projection_vector = tensor.set_subtensor(
            projection_vector[self.yksi_id], self.yksi_prob)
        self.kaksi_prob = 0.3
        projection_vector = tensor.set_subtensor(
            projection_vector[self.kaksi_id], self.kaksi_prob)
        self.eos_prob = 0.4
        projection_vector = tensor.set_subtensor(
            projection_vector[self.eos_id], self.eos_prob)
        self.unk_prob = 0.3
        projection_vector = tensor.set_subtensor(
            projection_vector[self.unk_id], self.unk_prob)
        self.network = DummyNetwork(self.vocabulary, projection_vector)

        lattice_path = os.path.join(script_path, 'lattice.slf')
        with open(lattice_path) as lattice_file:
            self.lattice = SLFLattice(lattice_file)

    def tearDown(self):
        """
        Tear down the next callable.

        Args:
            self: (todo): write your description
        """
        pass

    def test_copy_token(self):
        """
        Copy a token.

        Args:
            self: (todo): write your description
        """
        history = (1, 2, 3)
        token1 = LatticeDecoder.Token(history)
        token2 = LatticeDecoder.Token.copy(token1)
        token2.history = token2.history + (4,)
        self.assertSequenceEqual(token1.history, (1, 2, 3))
        self.assertSequenceEqual(token2.history, (1, 2, 3, 4))

    def test_recompute_hash(self):
        """
        Computes the hash of the given token.

        Args:
            self: (todo): write your description
        """
        token1 = LatticeDecoder.Token(history=(1, 12, 203, 3004, 23455))
        token2 = LatticeDecoder.Token(history=(2, 12, 203, 3004, 23455))
        token1.recompute_hash(None)
        token2.recompute_hash(None)
        self.assertNotEqual(token1.recombination_hash, token2.recombination_hash)
        token1.recompute_hash(5)
        token2.recompute_hash(5)
        self.assertNotEqual(token1.recombination_hash, token2.recombination_hash)
        token1.recompute_hash(4)
        token2.recompute_hash(4)
        self.assertEqual(token1.recombination_hash, token2.recombination_hash)

    def test_recompute_total(self):
        """
        Compute the total probability.

        Args:
            self: (todo): write your description
        """
        token = LatticeDecoder.Token(history=[1, 2],
                                     ac_logprob=math.log(0.1),
                                     lat_lm_logprob=math.log(0.2),
                                     nn_lm_logprob=math.log(0.3))
        token.recompute_total(0.25, 1.0, 0.0, True)
        # assert_almost_equal(token.lm_logprob,
        #                     math.log(0.25 * 0.3 + 0.75 * 0.2))
        assert_almost_equal(token.total_logprob,
                            math.log(0.1 * (0.25 * 0.3 + 0.75 * 0.2)))
        token.recompute_total(0.25, 1.0, 0.0, False)
        # assert_almost_equal(token.lm_logprob,
        #                     0.25 * math.log(0.3) + 0.75 * math.log(0.2))
        assert_almost_equal(token.total_logprob,
                            math.log(0.1) + 0.25 * math.log(0.3) + 0.75 * math.log(0.2))
        token.recompute_total(0.25, 10.0, 0.0, True)
        # assert_almost_equal(token.lm_logprob,
        #                     math.log(0.25 * 0.3 + 0.75 * 0.2))
        assert_almost_equal(token.total_logprob,
                            math.log(0.1) + math.log(0.25 * 0.3 + 0.75 * 0.2) * 10.0)
        token.recompute_total(0.25, 10.0, 0.0, False)
        # assert_almost_equal(token.lm_logprob,
        #                     0.25 * math.log(0.3) + 0.75 * math.log(0.2))
        assert_almost_equal(token.total_logprob,
                            math.log(0.1) + (0.25 * math.log(0.3) + 0.75 * math.log(0.2)) * 10.0)
        token.recompute_total(0.25, 10.0, -20.0, True)
        # assert_almost_equal(token.lm_logprob,
        #                     math.log(0.25 * 0.3 + 0.75 * 0.2))
        assert_almost_equal(token.total_logprob,
                            math.log(0.1) + math.log(0.25 * 0.3 + 0.75 * 0.2) * 10.0 - 20.0)
        token.recompute_total(0.25, 10.0, -20.0, False)
        # assert_almost_equal(token.lm_logprob,
        #                     0.25 * math.log(0.3) + 0.75 * math.log(0.2))
        assert_almost_equal(token.total_logprob,
                            math.log(0.1) + (0.25 * math.log(0.3) + 0.75 * math.log(0.2)) * 10.0 - 20.0)

        token = LatticeDecoder.Token(history=[1, 2],
                                     ac_logprob=-1000,
                                     lat_lm_logprob=-1001,
                                     nn_lm_logprob=-1002)
        token.recompute_total(0.75, 1.0, 0.0, True)
        # ln(exp(-1000) * (0.75 * exp(-1002) + 0.25 * exp(-1001)))
        assert_almost_equal(token.total_logprob, -2001.64263, decimal=4)

    def test_append_word(self):
        """
        Append a new word to the batch.

        Args:
            self: (todo): write your description
        """
        decoding_options = {
            'nnlm_weight': 1.0,
            'lm_scale': 1.0,
            'wi_penalty': 0.0,
            'unk_penalty': 0.0,
            'use_shortlist': False,
            'unk_from_lattice': False,
            'linear_interpolation': False,
            'max_tokens_per_node': 10,
            'beam': None,
            'recombination_order': None,
        }

        initial_state = RecurrentState(self.network.recurrent_state_size)
        token1 = LatticeDecoder.Token(history=(self.sos_id,), state=initial_state)
        token2 = LatticeDecoder.Token(history=(self.sos_id, self.yksi_id), state=initial_state)
        decoder = LatticeDecoder(self.network, decoding_options)

        self.assertSequenceEqual(token1.history, (self.sos_id,))
        self.assertSequenceEqual(token2.history, (self.sos_id, self.yksi_id))
        assert_equal(token1.state.get(0), numpy.zeros(shape=(1,1,3)).astype(theano.config.floatX))
        assert_equal(token2.state.get(0), numpy.zeros(shape=(1,1,3)).astype(theano.config.floatX))
        self.assertEqual(token1.nn_lm_logprob, 0.0)
        self.assertEqual(token2.nn_lm_logprob, 0.0)

        decoder._append_word([token1, token2], self.kaksi_id)
        self.assertSequenceEqual(token1.history, (self.sos_id, self.kaksi_id))
        self.assertSequenceEqual(token2.history, (self.sos_id, self.yksi_id, self.kaksi_id))
        assert_equal(token1.state.get(0), numpy.ones(shape=(1,1,3)).astype(theano.config.floatX))
        assert_equal(token2.state.get(0), numpy.ones(shape=(1,1,3)).astype(theano.config.floatX))
        token1_nn_lm_logprob = math.log(self.sos_prob + self.kaksi_prob)
        token2_nn_lm_logprob = math.log(self.yksi_prob + self.kaksi_prob)
        self.assertAlmostEqual(token1.nn_lm_logprob, token1_nn_lm_logprob)
        self.assertAlmostEqual(token2.nn_lm_logprob, token2_nn_lm_logprob)

        decoder._append_word([token1, token2], self.eos_id)
        self.assertSequenceEqual(token1.history, (self.sos_id, self.kaksi_id, self.eos_id))
        self.assertSequenceEqual(token2.history, (self.sos_id, self.yksi_id, self.kaksi_id, self.eos_id))
        assert_equal(token1.state.get(0), numpy.ones(shape=(1,1,3)).astype(theano.config.floatX) * 2)
        assert_equal(token2.state.get(0), numpy.ones(shape=(1,1,3)).astype(theano.config.floatX) * 2)
        token1_nn_lm_logprob += math.log(self.kaksi_prob + self.eos_prob)
        token2_nn_lm_logprob += math.log(self.kaksi_prob + self.eos_prob)
        self.assertAlmostEqual(token1.nn_lm_logprob, token1_nn_lm_logprob)
        self.assertAlmostEqual(token2.nn_lm_logprob, token2_nn_lm_logprob)

        lm_scale = 2.0
        token1.recompute_total(1.0, lm_scale, -0.01)
        token2.recompute_total(1.0, lm_scale, -0.01)
        self.assertAlmostEqual(token1.total_logprob, token1_nn_lm_logprob * lm_scale - 0.02)
        self.assertAlmostEqual(token2.total_logprob, token2_nn_lm_logprob * lm_scale - 0.03)

        decoding_options['use_shortlist'] = True
        decoder = LatticeDecoder(self.network, decoding_options)
        decoder._append_word([token1, token2], self.oos1_id)
        self.assertSequenceEqual(token1.history, [self.sos_id, self.kaksi_id, self.eos_id, self.oos1_id])
        self.assertSequenceEqual(token2.history, [self.sos_id, self.yksi_id, self.kaksi_id, self.eos_id, self.oos1_id])
        token1_nn_lm_logprob += math.log((self.eos_prob + self.unk_prob) / 3)
        token2_nn_lm_logprob += math.log((self.eos_prob + self.unk_prob) / 3)
        self.assertAlmostEqual(token1.nn_lm_logprob, token1_nn_lm_logprob)
        self.assertAlmostEqual(token2.nn_lm_logprob, token2_nn_lm_logprob)

        decoder._append_word([token1, token2], self.oos2_id)
        self.assertSequenceEqual(token1.history, [self.sos_id, self.kaksi_id, self.eos_id, self.oos1_id, self.oos2_id])
        self.assertSequenceEqual(token2.history, [self.sos_id, self.yksi_id, self.kaksi_id, self.eos_id, self.oos1_id, self.oos2_id])
        token1_nn_lm_logprob += math.log((self.unk_prob + self.unk_prob) / 3 * 2)
        token2_nn_lm_logprob += math.log((self.unk_prob + self.unk_prob) / 3 * 2)
        self.assertAlmostEqual(token1.nn_lm_logprob, token1_nn_lm_logprob)
        self.assertAlmostEqual(token2.nn_lm_logprob, token2_nn_lm_logprob)

    def test_prune(self):
        """
        Prune the hash of a decoder.

        Args:
            self: (todo): write your description
        """
        # token recombination
        decoder = DummyLatticeDecoder()
        decoder._max_tokens_per_node = None
        decoder._beam = None
        decoder._recombination_order = 3  # Not used.
        decoder._tokens[2][0].recombination_hash = 2
        decoder._tokens[2][1].recombination_hash = 2
        decoder._tokens[2][2].recombination_hash = 3
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 2)
        decoder._tokens[2][0].recombination_hash = 4
        decoder._tokens[2][1].recombination_hash = 4
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 1)

        # beam pruning
        decoder = DummyLatticeDecoder()
        decoder._max_tokens_per_node = None
        decoder._recombination_order = None
        # best_logprob = -20
        decoder._beam = 60
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 3)
        self.assertEqual(decoder._tokens[2][0].total_logprob, -30)
        self.assertEqual(decoder._tokens[2][1].total_logprob, -50)
        self.assertEqual(decoder._tokens[2][2].total_logprob, -70)
        decoder._beam = 50
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 2)
        self.assertEqual(decoder._tokens[2][0].total_logprob, -30)
        self.assertEqual(decoder._tokens[2][1].total_logprob, -50)
        decoder._beam = 31
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 2)
        self.assertEqual(decoder._tokens[2][0].total_logprob, -30)
        self.assertEqual(decoder._tokens[2][1].total_logprob, -50)
        decoder._beam = 15
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 1)
        self.assertEqual(decoder._tokens[2][0].total_logprob, -30)
        decoder._beam = 0
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 1)
        self.assertEqual(decoder._tokens[2][0].total_logprob, -30)

        # max tokens per node
        decoder = DummyLatticeDecoder()
        decoder._beam = None
        decoder._recombination_order = None
        decoder._max_tokens_per_node = 3
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 3)
        self.assertEqual(decoder._tokens[2][0].total_logprob, -30)
        self.assertEqual(decoder._tokens[2][1].total_logprob, -50)
        self.assertEqual(decoder._tokens[2][2].total_logprob, -70)
        decoder._max_tokens_per_node = 2
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 2)
        self.assertEqual(decoder._tokens[2][0].total_logprob, -30)
        self.assertEqual(decoder._tokens[2][1].total_logprob, -50)
        decoder._max_tokens_per_node = 1
        decoder._prune(decoder._sorted_nodes[2], decoder._sorted_nodes, decoder._tokens, [])
        self.assertEqual(len(decoder._tokens[2]), 1)
        self.assertEqual(decoder._tokens[2][0].total_logprob, -30)

    def test_decode(self):
        """
        Decodes the network.

        Args:
            self: (todo): write your description
        """
        vocabulary = Vocabulary.from_word_counts({
            'to': 1,
            'and': 1,
            'it': 1,
            'but': 1,
            'a.': 1,
            'in': 1,
            'a': 1,
            'at': 1,
            'the': 1,
            "didn't": 1,
            'elaborate': 1})
        projection_vector = tensor.ones(
            shape=(vocabulary.num_shortlist_words(),),
            dtype=theano.config.floatX)
        projection_vector *= 0.05
        network = DummyNetwork(vocabulary, projection_vector)

        decoding_options = {
            'nnlm_weight': 0.0,
            'lm_scale': None,
            'wi_penalty': None,
            'unk_penalty': None,
            'use_shortlist': False,
            'unk_from_lattice': False,
            'linear_interpolation': True,
            'max_tokens_per_node': None,
            'beam': None,
            'recombination_order': 20
        }
        decoder = LatticeDecoder(network, decoding_options)
        tokens = decoder.decode(self.lattice)[0]

        # Compare tokens to n-best list given by SRILM lattice-tool.
        log_scale = math.log(10)

        print()
        for token in tokens:
            print(token.ac_logprob / log_scale,
                  token.lat_lm_logprob / log_scale,
                  token.total_logprob / log_scale,
                  ' '.join(token.history_words(vocabulary)))

        all_paths = ["<s> it didn't elaborate </s>",
                     "<s> but it didn't elaborate </s>",
                     "<s> the didn't elaborate </s>",
                     "<s> and it didn't elaborate </s>",
                     "<s> e. didn't elaborate </s>",
                     "<s> in it didn't elaborate </s>",
                     "<s> a didn't elaborate </s>",
                     "<s> at it didn't elaborate </s>",
                     "<s> it it didn't elaborate </s>",
                     "<s> to it didn't elaborate </s>",
                     "<s> a. it didn't elaborate </s>",
                     "<s> a it didn't elaborate </s>"]
        paths = [' '.join(token.history_words(vocabulary)) for token in tokens]
        self.assertListEqual(paths, all_paths)

        token = tokens[0]
        history = ' '.join(token.history_words(vocabulary))
        self.assertAlmostEqual(token.ac_logprob / log_scale, -8686.28, places=2)
        self.assertAlmostEqual(token.lat_lm_logprob / log_scale, -94.3896, places=2)
        self.assertAlmostEqual(token.nn_lm_logprob, math.log(0.1) * 4)

        token = tokens[1]
        self.assertAlmostEqual(token.ac_logprob / log_scale, -8743.96, places=2)
        self.assertAlmostEqual(token.lat_lm_logprob / log_scale, -111.488, places=2)
        self.assertAlmostEqual(token.nn_lm_logprob, math.log(0.1) * 5)

        token = tokens[-1]
        self.assertAlmostEqual(token.ac_logprob / log_scale, -8696.26, places=2)
        self.assertAlmostEqual(token.lat_lm_logprob / log_scale, -178.00, places=2)
        self.assertAlmostEqual(token.nn_lm_logprob, math.log(0.1) * 5)

if __name__ == '__main__':
    unittest.main()
