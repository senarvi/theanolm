#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy
from numpy.testing import assert_equal, assert_almost_equal

from theanolm.network.basiclayer import BasicLayer

class DummyParameters(object):
    def __init__(self):
        """
        Initialize device properties.

        Args:
            self: (todo): write your description
        """
        self._vars = dict()
        self._devs = dict()

    def __getitem__(self, path):
        """
        Return the value of a given path.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        return self._vars[path]

    def add(self, path, value, device=None):
        """
        Add a device path.

        Args:
            self: (todo): write your description
            path: (str): write your description
            value: (todo): write your description
            device: (int): write your description
        """
        self._vars[path] = value
        self._devs[path] = device

    def get_device(self, path):
        """
        Get the device.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        return self._devs[path]

class DummyLayer(BasicLayer):
    def __init__(self, layer_options):
        """
        Initialize layer_options.

        Args:
            self: (todo): write your description
            layer_options: (todo): write your description
        """
        super().__init__(layer_options, None)
        self._params = DummyParameters()

    def create_structure(self):
        """
        Creates a structure.

        Args:
            self: (todo): write your description
        """
        pass

class TestBasicLayer(unittest.TestCase):
    def setUp(self):
        """
        Set layer options.

        Args:
            self: (todo): write your description
        """
        self.layer_options = dict()
        self.layer_options['name'] = 'layer_name'
        self.layer_options['input_layers'] = []
        self.layer_options['devices'] = ['dev0', 'dev1', 'dev2']

    def tearDown(self):
        """
        Tear down the next callable.

        Args:
            self: (todo): write your description
        """
        pass

    def test_param_path(self):
        """
        Set the path of the layer.

        Args:
            self: (todo): write your description
        """
        layer = DummyLayer(self.layer_options)
        self.assertEqual(layer._param_path('var1'), 'layers/layer_name/var1')
        self.assertEqual(layer._param_path('var2', 'dev1'), 'layers/layer_name/var2/dev1')

    def test_get_param(self):
        """
        Method to make a new test layer.

        Args:
            self: (todo): write your description
        """
        layer = DummyLayer(self.layer_options)
        layer._params.add('layers/layer_name/var1', 1)
        layer._params.add('layers/layer_name/var2', 2)
        layer._params.add('layers/layer_name/var3/dev1', 3)
        layer._params.add('layers/layer_name/var3/dev2', 4)
        self.assertEqual(layer._get_param('var1'), 1)
        self.assertEqual(layer._get_param('var2'), 2)
        self.assertEqual(layer._get_param('var3', 'dev1'), 3)
        self.assertEqual(layer._get_param('var3', 'dev2'), 4)

    def test_init_weight(self):
        """
        Initialize the weights.

        Args:
            self: (todo): write your description
        """
        layer = DummyLayer(self.layer_options)

        # standard normal distribution
        layer._init_weight('weight1', (100, 200), split_to_devices=False)
        weight = layer._get_param('weight1')
        self.assertListEqual(list(weight.shape), [100, 200])
        self.assertLess(numpy.min(weight), -1.0)
        self.assertGreater(numpy.max(weight), 1.0)

        # scaled
        layer._init_weight('weight2', (10, 20), 0.001, split_to_devices=False)
        weight = layer._get_param('weight2')
        self.assertGreater(numpy.min(weight), -1.0)
        self.assertLess(numpy.max(weight), 1.0)

        # orthogonal
        layer._init_weight('weight3', (100, 100), 0.001, split_to_devices=False)
        weight = layer._get_param('weight3')
        assert_almost_equal(numpy.dot(weight, weight.T), numpy.identity(100))

        # tiled
        layer._init_weight('weight4', (100, 200), count=3,
                           split_to_devices=False)
        weight = layer._get_param('weight4')
        self.assertListEqual(list(weight.shape), [100, 600])

        # split
        layer._init_weight('weight5', (100, 600), split_to_devices=True)
        weight = layer._get_param('weight5', 'dev0')
        self.assertListEqual(list(weight.shape), [100, 200])
        weight = layer._get_param('weight5', 'dev1')
        self.assertListEqual(list(weight.shape), [100, 200])
        weight = layer._get_param('weight5', 'dev2')
        self.assertListEqual(list(weight.shape), [100, 200])
        self.assertEqual(layer._params.get_device('layers/layer_name/weight5/dev0'), 'dev0')
        self.assertEqual(layer._params.get_device('layers/layer_name/weight5/dev1'), 'dev1')
        self.assertEqual(layer._params.get_device('layers/layer_name/weight5/dev2'), 'dev2')

        layer._init_weight('weight6', (100, 600), count=2, split_to_devices=True)
        weight = layer._get_param('weight6', 'dev0')
        self.assertListEqual(list(weight.shape), [100, 400])
        weight = layer._get_param('weight6', 'dev1')
        self.assertListEqual(list(weight.shape), [100, 400])
        weight = layer._get_param('weight6', 'dev2')
        self.assertListEqual(list(weight.shape), [100, 400])
        self.assertEqual(layer._params.get_device('layers/layer_name/weight6/dev0'), 'dev0')
        self.assertEqual(layer._params.get_device('layers/layer_name/weight6/dev1'), 'dev1')
        self.assertEqual(layer._params.get_device('layers/layer_name/weight6/dev2'), 'dev2')

    def test_init_bias(self):
        """
        Test for the bias.

        Args:
            self: (todo): write your description
        """
        layer = DummyLayer(self.layer_options)

        # zeros
        layer._init_bias('bias1', (10, 20), split_to_devices=False)
        bias = layer._get_param('bias1')
        assert_equal(bias, numpy.zeros((10, 20)))

        # single value
        layer._init_bias('bias2', (10, 20), 5, split_to_devices=False)
        bias = layer._get_param('bias2')
        assert_equal(bias, numpy.ones((10, 20)) * 5)

        # array
        layer._init_bias('bias3', (10, 20),
                         numpy.arange(10 * 20).reshape(10, 20),
                         split_to_devices=False)
        bias = layer._get_param('bias3')
        assert_equal(bias, numpy.arange(10 * 20).reshape(10, 20))

        # tiled
        layer._init_bias('bias4', (3, 2), value=[5, 6, 7],
                         split_to_devices=False)
        bias = layer._get_param('bias4')
        value = numpy.array([[5, 5, 6, 6, 7, 7],
                             [5, 5, 6, 6, 7, 7],
                             [5, 5, 6, 6, 7, 7]])
        assert_almost_equal(bias, value)

        # split
        layer._init_bias('bias5', (2, 3), value=[5, 6, 7],
                         split_to_devices=True)
        bias = layer._get_param('bias5', 'dev0')
        value = numpy.ones((2, 3)) * 5
        assert_almost_equal(bias, value)
        bias = layer._get_param('bias5', 'dev1')
        value = numpy.ones((2, 3)) * 6
        assert_almost_equal(bias, value)
        bias = layer._get_param('bias5', 'dev2')
        value = numpy.ones((2, 3)) * 7
        assert_almost_equal(bias, value)
        self.assertEqual(layer._params.get_device('layers/layer_name/bias5/dev0'), 'dev0')
        self.assertEqual(layer._params.get_device('layers/layer_name/bias5/dev1'), 'dev1')
        self.assertEqual(layer._params.get_device('layers/layer_name/bias5/dev2'), 'dev2')

    def test_split_per_device(self):
        """
        Split the device splits.

        Args:
            self: (todo): write your description
        """
        layer = DummyLayer(self.layer_options)
        ranges = layer._split_per_device(10)
        self.assertEqual(len(ranges), 3)
        self.assertTrue(len(ranges[0]) == 3 or len(ranges[0]) == 4)
        self.assertTrue(len(ranges[1]) == 3 or len(ranges[1]) == 4)
        self.assertTrue(len(ranges[2]) == 3 or len(ranges[2]) == 4)
        self.assertEqual(sum(len(x) for x in ranges), 10)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[1][0], ranges[0][-1] + 1)
        self.assertEqual(ranges[2][0], ranges[1][-1] + 1)
        self.assertEqual(ranges[2][-1], 9)

if __name__ == '__main__':
    unittest.main()
