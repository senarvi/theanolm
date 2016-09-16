#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import math
import numpy
from numpy.testing import assert_equal
from theanolm.network import RecurrentState

class TestRecurrentState(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        state = RecurrentState([200, 100, 300], 3)
        self.assertEqual(len(state.get()), 3)
        self.assertEqual(state.get(0).shape, (1,3,200))
        self.assertEqual(state.get(1).shape, (1,3,100))
        self.assertEqual(state.get(2).shape, (1,3,300))
        assert_equal(state.get(0), numpy.zeros(shape=(1,3,200), dtype='int64'))
        assert_equal(state.get(1), numpy.zeros(shape=(1,3,100), dtype='int64'))
        assert_equal(state.get(2), numpy.zeros(shape=(1,3,300), dtype='int64'))

        layer1_state = numpy.arange(15, dtype='int64').reshape((1, 3, 5))
        layer2_state = numpy.arange(30, dtype='int64').reshape((1, 3, 10))
        state = RecurrentState([5, 10], 3, [layer1_state, layer2_state])
        assert_equal(state.get(0), layer1_state)
        assert_equal(state.get(1), layer2_state)

    def test_set(self):
        state = RecurrentState([5, 10], 3)
        layer1_state = numpy.arange(15, dtype='int64').reshape((1, 3, 5))
        layer2_state = numpy.arange(30, dtype='int64').reshape((1, 3, 10))
        state.set([layer1_state, layer2_state])
        assert_equal(state.get(0), layer1_state)
        assert_equal(state.get(1), layer2_state)

        with self.assertRaises(ValueError):
            state.set([layer2_state, layer1_state])

    def test_combine_sequences(self):
        state1 = RecurrentState([5, 10], 1)
        layer1_state = numpy.arange(5, dtype='int64').reshape(1, 1, 5)
        layer2_state = numpy.arange(10, 20, dtype='int64').reshape(1, 1, 10)
        state1.set([layer1_state, layer2_state])

        state2 = RecurrentState([5, 10], 1)
        layer1_state = numpy.arange(100, 105, dtype='int64').reshape(1, 1, 5)
        layer2_state = numpy.arange(110, 120, dtype='int64').reshape(1, 1, 10)
        state2.set([layer1_state, layer2_state])

        state3 = RecurrentState([5, 10], 2)
        layer1_state = numpy.arange(200, 210, dtype='int64').reshape(1, 2, 5)
        layer2_state = numpy.arange(210, 230, dtype='int64').reshape(1, 2, 10)
        state3.set([layer1_state, layer2_state])

        combined_state = RecurrentState.combine_sequences([state1, state2, state3])
        self.assertEqual(combined_state.num_sequences, 4)
        self.assertEqual(len(combined_state.get()), 2)
        self.assertEqual(combined_state.get(0).shape, (1,4,5))
        self.assertEqual(combined_state.get(1).shape, (1,4,10))
        assert_equal(combined_state.get(0), numpy.asarray(
            [[list(range(5)),
              list(range(100, 105)),
              list(range(200, 205)),
              list(range(205, 210))]],
            dtype='int64'))
        assert_equal(combined_state.get(1), numpy.asarray(
            [[list(range(10, 20)),
              list(range(110, 120)),
              list(range(210, 220)),
              list(range(220, 230))]],
            dtype='int64'))

        state4 = RecurrentState([5, 11], 2)
        with self.assertRaises(ValueError):
            combined_state = RecurrentState.combine_sequences([state1, state2, state3, state4])

if __name__ == '__main__':
    unittest.main()
