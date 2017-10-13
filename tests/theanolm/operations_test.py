#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy
import theano
import theano.tensor as tensor

from theanolm.backend import conv1d, conv2d

class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_conv1d(self):
        x_tensor = tensor.tensor3("x")
        filter_tensor = tensor.tensor3("filter")
        y_tensor = conv1d(x_tensor, filter_tensor)
        f = theano.function([x_tensor, filter_tensor], y_tensor)

        num_samples = 5
        num_elements = 10
        num_features = 2
        num_filters = 2
        x = numpy.arange(num_samples * num_elements * num_features)
        x = x.reshape([num_samples, num_elements, num_features])
        # Filter 1 is applied on feature 2, filter 2 is applied on feature 1.
        # Filter 1: [1]  Filter 2: [0]
        #           [0]            [1]
        #           [1]            [0]
        filters = numpy.array([[[0, 0],
                                [1, 0]],
                               [[0, 1],
                                [0, 0]],
                               [[0, 0],
                                [1, 0]]])

        y = f(x, filters)

        self.assertEqual(y.shape[0], num_samples)
        self.assertEqual(y.shape[1], num_elements - 2)
        self.assertEqual(y.shape[2], num_filters)

        for sample in range(num_samples):
            feature1_middle = (sample * 10 + numpy.arange(1, 9)) * 2
            feature2_left = (sample * 10 + numpy.arange(0, 8)) * 2 + 1
            feature2_right = (sample * 10 + numpy.arange(2, 10)) * 2 + 1
            self.assertSequenceEqual(list(y[sample, :, 0]),
                                     list(feature2_left + feature2_right))
            self.assertSequenceEqual(list(y[sample, :, 1]),
                                     list(feature1_middle))

    def test_conv2d(self):
        x_tensor = tensor.tensor4("x")
        filter_tensor = tensor.tensor4("filter")
        y_tensor = conv2d(x_tensor, filter_tensor, padding="same")
        f = theano.function([x_tensor, filter_tensor], y_tensor)

        num_samples = 5
        num_rows = 10
        num_columns = 10
        num_features = 2
        x = numpy.ones([num_samples, num_rows, num_columns, num_features])
        filters = numpy.zeros([3, 3, num_features, 7])

        y = f(x, filters)

        self.assertEqual(y.shape[0], num_samples)
        self.assertEqual(y.shape[1], num_rows)
        self.assertEqual(y.shape[2], num_columns)
        self.assertEqual(y.shape[3], 7)

if __name__ == "__main__":
    unittest.main()