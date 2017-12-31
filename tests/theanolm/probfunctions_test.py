#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import math

from theanolm.backend.probfunctions import *

class TestProbFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_interpolate_linear(self):
        self.assertAlmostEqual(
            interpolate_linear(numpy.log(0.2), numpy.log(0.3), numpy.float(0.25)),
            numpy.log(0.25 * 0.2 + 0.75 * 0.3))
        self.assertAlmostEqual(
            interpolate_linear(numpy.float('-inf'), numpy.log(0.3), numpy.float(0.01)),
            numpy.log(0.3 * 0.99))
        self.assertEqual(
            interpolate_linear(numpy.float('-inf'), numpy.float(-10), numpy.float(0)),
            numpy.float(-10))
        self.assertAlmostEqual(
            interpolate_linear(numpy.log(0.3), numpy.float('-inf'), numpy.float(0.99)),
            numpy.log(0.3 * 0.99))
        self.assertEqual(
            interpolate_linear(numpy.float(-10), numpy.float('-inf'), numpy.float(1)),
            numpy.float(-10))
        self.assertAlmostEqual(
            interpolate_linear(numpy.float(-1001), numpy.float(-1002), numpy.float(0.25)),
            numpy.float(-1001.64263),  # ln(0.25 * exp(-1001) + 0.75 * exp(-1002))
            places=4)

    def test_interpolate_loglinear(self):
        self.assertEqual(
            interpolate_loglinear(numpy.float(-1001), numpy.float(-1002),
                                  numpy.float(0.25), numpy.float(0.75)),
            numpy.float(-1001.75))
        self.assertEqual(
            interpolate_loglinear(numpy.float('-inf'), numpy.float(-1002),
                                  numpy.float(0.25), numpy.float(0.75)),
            numpy.float('-inf'))
        self.assertEqual(
            interpolate_loglinear(numpy.float('-inf'), numpy.float(-1002),
                                  numpy.float(0), numpy.float(1)),
            numpy.float(-1002))
        self.assertEqual(
            interpolate_loglinear(numpy.float(-1001), numpy.float('-inf'),
                                  numpy.float(0.25), numpy.float(0.75)),
            numpy.float('-inf'))
        self.assertEqual(
            interpolate_loglinear(numpy.float(-1001), numpy.float('-inf'),
                                  numpy.float(1), numpy.float(0)),
            numpy.float(-1001))

if __name__ == '__main__':
    unittest.main()
