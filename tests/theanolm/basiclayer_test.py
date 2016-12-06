#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import mmap
import numpy
from numpy.testing import assert_equal
import theanolm
from theanolm.network.basiclayer import BasicLayer

class DummyLayer(BasicLayer):
    def __init__(self, layer_options):
        super().__init__(layer_options, None)

    def create_structure(self):
        pass

class TestBasicLayer(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_size_per_device(self):
        layer_options = dict()
        layer_options['name'] = 'layer_name'
        layer_options['input_layers'] = []
        layer_options['devices'] = ['dev0', 'dev1', 'dev2']
        layer = DummyLayer(layer_options)
        sizes = layer._size_per_device(10)
        self.assertEqual(sum(sizes), 10)
        self.assertEqual(len(sizes), 3)
        self.assertTrue(sizes[0] == 3 or sizes[0] == 4)
        self.assertTrue(sizes[1] == 3 or sizes[1] == 4)
        self.assertTrue(sizes[2] == 3 or sizes[2] == 4)

if __name__ == '__main__':
    unittest.main()
