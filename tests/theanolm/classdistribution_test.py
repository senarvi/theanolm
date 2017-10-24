#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theanolm.backend import UniformDistribution, LogUniformDistribution
from theanolm.backend import MultinomialDistribution

class TestClassDistribution(unittest.TestCase):
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

    def test_uniform_distribution_sample(self):
        distribution = UniformDistribution(self.random, 100)
        sample_tensor = distribution.sample(200, 400)
        f = theano.function([], [sample_tensor])

        sample = f()[0]
        self.assertEqual(sample.ndim, 2)
        self.assertEqual(sample.shape[0], 200)
        self.assertEqual(sample.shape[1], 400)
        self.assertEqual(sample.dtype, 'int64')
        self.assertTrue(numpy.all(numpy.greater_equal(sample, 0)))
        self.assertTrue(numpy.all(numpy.less(sample, 100)))

    def test_uniform_distribution_probs(self):
        distribution = UniformDistribution(self.random, 10)
        x_tensor = tensor.vector(dtype='int64')
        x_tensor.tag.test_value = numpy.array([1, 3, 5])
        probs = distribution.probs(x_tensor)
        self.assertEqual(probs, 0.1)

    def test_log_uniform_distribution_sample(self):
        distribution = LogUniformDistribution(self.random, 100)
        sample_tensor = distribution.sample(200, 400)
        f = theano.function([], [sample_tensor])

        sample = f()[0]
        self.assertEqual(sample.ndim, 2)
        self.assertEqual(sample.shape[0], 200)
        self.assertEqual(sample.shape[1], 400)
        self.assertEqual(sample.dtype, 'int64')
        self.assertTrue(numpy.all(numpy.greater_equal(sample, 0)))
        self.assertTrue(numpy.all(numpy.less(sample, 100)))

    def test_log_uniform_distribution_probs(self):
        distribution = LogUniformDistribution(self.random, 10)
        x_tensor = tensor.vector(dtype='int64')
        x_tensor.tag.test_value = numpy.array([1, 3, 5])
        probs_tensor = distribution.probs(x_tensor)
        f = theano.function([x_tensor], [probs_tensor])

        x = numpy.arange(10)
        probs = f(x)[0]
        self.assertAlmostEqual(probs.sum(), 1.0)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])
        self.assertGreater(probs[2], probs[3])
        self.assertGreater(probs[3], probs[4])
        self.assertGreater(probs[4], probs[5])
        self.assertGreater(probs[5], probs[6])
        self.assertGreater(probs[6], probs[7])
        self.assertGreater(probs[7], probs[8])
        self.assertGreater(probs[8], probs[9])
        self.assertAlmostEqual(probs[0], numpy.log(2) / numpy.log(11))
        self.assertAlmostEqual(probs[2:5].sum(), (numpy.log(6) - numpy.log(3)) / numpy.log(11))

    def test_multinomial_distribution_sample(self):
        probs = numpy.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001])
        probs /= probs.sum()
        probs_tensor = theano.shared(probs)
        distribution = MultinomialDistribution(self.random, probs_tensor)
        sample_tensor = distribution.sample(200, 3)
        f = theano.function([], [sample_tensor])

        sample = f()[0]
        self.assertEqual(sample.ndim, 2)
        self.assertEqual(sample.shape[0], 200)
        self.assertEqual(sample.shape[1], 3)
        self.assertEqual(sample.dtype, 'int64')
        self.assertTrue(numpy.all(numpy.greater_equal(sample, 0)))
        self.assertTrue(numpy.all(numpy.less(sample, 10)))

    def test_multinomial_distribution_probs(self):
        probs = numpy.array([0.01, 0.02, 0.03, 0.04, 0.1, 0.3, 0.5])
        probs_tensor = theano.shared(probs)
        distribution = MultinomialDistribution(self.random, probs_tensor)
        x_tensor = tensor.vector(dtype='int64')
        x_tensor.tag.test_value = numpy.array([1, 3, 5])
        probs_tensor = distribution.probs(x_tensor)
        f = theano.function([x_tensor], [probs_tensor])

        x = numpy.arange(7)
        probs = f(x)[0]
        self.assertAlmostEqual(probs[0], 0.01)
        self.assertAlmostEqual(probs[1], 0.02)
        self.assertAlmostEqual(probs[2], 0.03)
        self.assertAlmostEqual(probs[3], 0.04)
        self.assertAlmostEqual(probs[4], 0.1)
        self.assertAlmostEqual(probs[5], 0.3)
        self.assertAlmostEqual(probs[6], 0.5)

if __name__ == '__main__':
    unittest.main()
