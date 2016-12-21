#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import math
from theanolm.probfunctions import *

class TestProbFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_interpolate_linear(self):
        self.assertAlmostEqual(
            interpolate_linear(math.log(0.2), math.log(0.3), 0.25),
            math.log(0.25 * 0.2 + 0.75 * 0.3))
        self.assertAlmostEqual(
            interpolate_linear(float('-inf'), math.log(0.3), 0.01),
            math.log(0.3 * 0.99))
        self.assertEqual(
            interpolate_linear(float('-inf'), -10.0, 0.0),
            -10.0)
        self.assertAlmostEqual(
            interpolate_linear(math.log(0.3), float('-inf'), 0.99),
            math.log(0.3 * 0.99))
        self.assertEqual(
            interpolate_linear(-10.0, float('-inf'), 1.0),
            -10.0)
        self.assertAlmostEqual(
            interpolate_linear(-1001, -1002, 0.25),
            -1001.64263,  # ln(0.25 * exp(-1001) + 0.75 * exp(-1002))
            places=4)

    def test_interpolate_loglinear(self):
        self.assertEqual(
            interpolate_loglinear(-1001.0, -1002.0, 0.25, 0.75),
            -1001.75)
        self.assertEqual(
            interpolate_loglinear(float('-inf'), -1002.0, 0.25, 0.75),
            float('-inf'))
        self.assertEqual(
            interpolate_loglinear(float('-inf'), -1002.0, 0.0, 1.0),
            -1002.0)
        self.assertEqual(
            interpolate_loglinear(-1001.0, float('-inf'), 0.25, 0.75),
            float('-inf'))
        self.assertEqual(
            interpolate_loglinear(-1001.0, float('-inf'), 1.0, 0.0),
            -1001.0)

if __name__ == '__main__':
    unittest.main()
