#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy
from numpy.testing import assert_equal, assert_almost_equal
from theanolm.network.basiclayer import BasicLayer

class DummyParameters(object):
    def __init__(self):
        self._vars = dict()
        self._devs = dict()

    def __getitem__(self, path):
        return self._vars[path]

    def add(self, path, value, device=None):
        self._vars[path] = value
        self._devs[path] = device

    def get_device(self, path):
        return self._devs[path]

class DummyLayer(BasicLayer):
    def __init__(self, layer_options):
        super().__init__(layer_options, None)
        self.params = DummyParameters()

    def create_structure(self):
        pass

class TestBasicLayer(unittest.TestCase):
    def setUp(self):
        self.layer_options = dict()
        self.layer_options['name'] = 'layer_name'
        self.layer_options['input_layers'] = []
        self.layer_options['devices'] = ['dev0', 'dev1', 'dev2']

    def tearDown(self):
        pass

    def test_param_path(self):
        layer = DummyLayer(self.layer_options)
        self.assertEqual(layer._param_path('var1'), 'layers/layer_name/var1')
        self.assertEqual(layer._param_path('var2', 'dev1'), 'layers/layer_name/var2/dev1')

    def test_get_param(self):
        layer = DummyLayer(self.layer_options)
        layer.params.add('layers/layer_name/var1', 1)
        layer.params.add('layers/layer_name/var2', 2)
        layer.params.add('layers/layer_name/var3/dev1', 3)
        layer.params.add('layers/layer_name/var3/dev2', 4)
        self.assertEqual(layer._get_param('var1'), 1)
        self.assertEqual(layer._get_param('var2'), 2)
        self.assertEqual(layer._get_param('var3', 'dev1'), 3)
        self.assertEqual(layer._get_param('var3', 'dev2'), 4)

    def test_init_weight(self):
        layer = DummyLayer(self.layer_options)

        # standard normal distribution
        layer._init_weight('weight1', (100, 200))
        weight = layer._get_param('weight1')
        self.assertListEqual(list(weight.shape), [100, 200])
        self.assertLess(numpy.min(weight), -1.0)
        self.assertGreater(numpy.max(weight), 1.0)

        # scaled
        layer._init_weight('weight2', (10, 20), 0.001)
        weight = layer._get_param('weight2')
        self.assertGreater(numpy.min(weight), -1.0)
        self.assertLess(numpy.max(weight), 1.0)

        # orthogonal
        layer._init_weight('weight3', (100, 100), 0.001)
        weight = layer._get_param('weight3')
        assert_almost_equal(numpy.dot(weight, weight.T), numpy.identity(100))

        # tiled
        layer._init_weight('weight4', (100, 200), count=3)
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
        self.assertEqual(layer.params.get_device('layers/layer_name/weight5/dev0'), 'dev0')
        self.assertEqual(layer.params.get_device('layers/layer_name/weight5/dev1'), 'dev1')
        self.assertEqual(layer.params.get_device('layers/layer_name/weight5/dev2'), 'dev2')

        layer._init_weight('weight6', (100, 600), count=2, split_to_devices=True)
        weight = layer._get_param('weight6', 'dev0')
        self.assertListEqual(list(weight.shape), [100, 400])
        weight = layer._get_param('weight6', 'dev1')
        self.assertListEqual(list(weight.shape), [100, 400])
        weight = layer._get_param('weight6', 'dev2')
        self.assertListEqual(list(weight.shape), [100, 400])
        self.assertEqual(layer.params.get_device('layers/layer_name/weight6/dev0'), 'dev0')
        self.assertEqual(layer.params.get_device('layers/layer_name/weight6/dev1'), 'dev1')
        self.assertEqual(layer.params.get_device('layers/layer_name/weight6/dev2'), 'dev2')

    def test_init_bias(self):
        layer = DummyLayer(self.layer_options)

        # zeros
        layer._init_bias('bias1', (10, 20))
        bias = layer._get_param('bias1')
        assert_equal(bias, numpy.zeros((10, 20)))

        # single value
        layer._init_bias('bias2', (10, 20), 5)
        bias = layer._get_param('bias2')
        assert_equal(bias, numpy.ones((10, 20)) * 5)

        # array
        layer._init_bias('bias3', (10, 20), numpy.arange(10 * 20).reshape(10, 20))
        bias = layer._get_param('bias3')
        assert_equal(bias, numpy.arange(10 * 20).reshape(10, 20))

        # tiled
        layer._init_bias('biasr', (3, 2), value=[5, 6, 7])
        bias = layer._get_param('biasr')
        value = numpy.array([[5, 5, 6, 6, 7, 7],
                             [5, 5, 6, 6, 7, 7],
                             [5, 5, 6, 6, 7, 7]])
        assert_almost_equal(bias, value)

        # split
        layer._init_bias('bias5', (2, 3), value=[5, 6, 7], split_to_devices=True)
        bias = layer._get_param('bias5', 'dev0')
        value = numpy.array([[5, 6, 7],
                             [5, 6, 7]])
        assert_almost_equal(bias, value)
        bias = layer._get_param('bias5', 'dev1')
        assert_almost_equal(bias, value)
        bias = layer._get_param('bias5', 'dev2')
        assert_almost_equal(bias, value)
        self.assertEqual(layer.params.get_device('layers/layer_name/bias5/dev0'), 'dev0')
        self.assertEqual(layer.params.get_device('layers/layer_name/bias5/dev1'), 'dev1')
        self.assertEqual(layer.params.get_device('layers/layer_name/bias5/dev2'), 'dev2')

    def test_split_to_devices(self):
        layer = DummyLayer(self.layer_options)
        value = numpy.concatenate([numpy.ones((10, 6)) * 5,
                                   numpy.ones((10, 6)) * 6,
                                   numpy.ones((10, 6)) * 7],
                                  axis=1)
        layer._split_to_devices('layers/layer_name/var', value, 6)
        part_value = numpy.concatenate([numpy.ones((10, 2)) * 5,
                                        numpy.ones((10, 2)) * 6,
                                        numpy.ones((10, 2)) * 7],
                                       axis=1)
        dev0_value = layer._get_param('var', 'dev0')
        dev1_value = layer._get_param('var', 'dev1')
        dev2_value = layer._get_param('var', 'dev2')
        assert_equal(dev0_value, part_value)
        assert_equal(dev1_value, part_value)
        assert_equal(dev2_value, part_value)

    def test_size_per_device(self):
        layer = DummyLayer(self.layer_options)
        sizes = layer._size_per_device(10)
        self.assertEqual(sum(sizes), 10)
        self.assertEqual(len(sizes), 3)
        self.assertTrue(sizes[0] == 3 or sizes[0] == 4)
        self.assertTrue(sizes[1] == 3 or sizes[1] == 4)
        self.assertTrue(sizes[2] == 3 or sizes[2] == 4)

if __name__ == '__main__':
    unittest.main()
