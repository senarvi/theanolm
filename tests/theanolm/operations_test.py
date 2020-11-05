#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy
import theano
import theano.tensor as tensor

from theanolm.backend import conv1d, conv2d
from theanolm.backend import l1_norm, sum_of_squares

class Test(unittest.TestCase):
    def setUp(self):
        """
        Sets the result of this thread.

        Args:
            self: (todo): write your description
        """
        pass

    def tearDown(self):
        """
        Tear down the next callable.

        Args:
            self: (todo): write your description
        """
        pass

    def test_conv1d(self):
        """
        Test for a 2d tensor.

        Args:
            self: (todo): write your description
        """
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
        """
        Convert 2d tensor to 2d tensor.

        Args:
            self: (todo): write your description
        """
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

    def test_l1_norm(self):
        """
        Test if the norm of the l1 norm.

        Args:
            self: (todo): write your description
        """
        a_tensor = tensor.matrix("a")
        b_tensor = tensor.matrix("b")
        y_tensor = l1_norm([a_tensor, b_tensor])
        f = theano.function([a_tensor, b_tensor], y_tensor)

        a = numpy.arange(9).reshape(3, 3)
        b = numpy.array([[-9, 10], [11, -12]])
        y = f(a, b)

        self.assertEqual(y, numpy.arange(13).sum())

    def test_sum_of_squares(self):
        """
        Test the sum of the sum of the tensors.

        Args:
            self: (todo): write your description
        """
        a_tensor = tensor.matrix("a")
        b_tensor = tensor.matrix("b")
        y_tensor = sum_of_squares([a_tensor, b_tensor])
        f = theano.function([a_tensor, b_tensor], y_tensor)

        a = numpy.arange(9).reshape(3, 3)
        b = numpy.array([[9, -10], [-11, 12]])
        y = f(a, b)

        self.assertEqual(y, numpy.square(numpy.arange(13)).sum())

if __name__ == "__main__":
    unittest.main()
