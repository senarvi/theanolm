#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theanolm.network.noisesampler import *

class TestNoiseSampler(unittest.TestCase):
    def setUp(self):
        theano.config.compute_test_value = 'warn'

        M1 = 2147483647
        M2 = 2147462579
        random_seed = [
            numpy.random.randint(0, M1),
            numpy.random.randint(0, M1),
            numpy.random.randint(1, M1),
            numpy.random.randint(0, M2),
            numpy.random.randint(0, M2),
            numpy.random.randint(1, M2)]
        self.random = RandomStreams(random_seed)

    def tearDown(self):
        pass

    def test_uniform_sampler(self):
        return
        sampler = UniformSampler(self.random, 100)
        tensor = sampler.sample(200, 400)
        f = theano.function([], [tensor])
        sample = f()[0]
        self.assertEqual(sample.ndim, 2)
        self.assertEqual(sample.shape[0], 200)
        self.assertEqual(sample.shape[1], 400)
        self.assertEqual(sample.dtype, 'int64')
        self.assertTrue(numpy.all(numpy.greater_equal(sample, 0)))
        self.assertTrue(numpy.all(numpy.less(sample, 100)))

    def test_log_uniform_sampler(self):
        return
        sampler = LogUniformSampler(self.random, 100)
        tensor = sampler.sample(200, 400)
        f = theano.function([], [tensor])
        sample = f()[0]
        self.assertEqual(sample.ndim, 2)
        self.assertEqual(sample.shape[0], 200)
        self.assertEqual(sample.shape[1], 400)
        self.assertEqual(sample.dtype, 'int64')
        self.assertTrue(numpy.all(numpy.greater_equal(sample, 0)))
        self.assertTrue(numpy.all(numpy.less(sample, 100)))

    def test_multinomial_sampler(self):
        return
        probs = numpy.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001])
        probs_tensor = theano.shared(probs)
        sampler = LogUniformSampler(self.random, probs_tensor)
        sample_tensor = sampler.sample(200, 400)
        f = theano.function([], [sample_tensor])
        sample = f()[0]
        self.assertEqual(sample.ndim, 2)
        self.assertEqual(sample.shape[0], 200)
        self.assertEqual(sample.shape[1], 400)
        self.assertEqual(sample.dtype, 'int64')
        self.assertTrue(numpy.all(numpy.greater_equal(sample, 0)))
        self.assertTrue(numpy.all(numpy.less(sample, 10)))

if __name__ == '__main__':
    unittest.main()
