#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from collections import OrderedDict
import theano.tensor as tensor

from matrixfunctions import normalized_weight

class SkipLayer(object):
	""" Skip-Layer
	
	A skip-layer combines two inputs by addition (in our case word projections
	with hidden layer outputs).
	"""

	def __init__(self, in1_size, in2_size, out_size):
		"""Initializes the parameters for a skip-layer of a neural network.

		:type options: dict
		:param options: a dictionary of training options

		:type in1_size: int
		:param options: number of connections from the first input layer 
	
		:type in2_size: int
		:param options: number of connections from the second input layer
	
		:type out_size: int
		:param options: number of output connections
		"""

		# Create the parameters.
		self.init_params = OrderedDict()

		self.init_params['skip_W_in1'] = \
				normalized_weight(in1_size, out_size, scale=0.01, ortho=False)

		self.init_params['skip_b_in1'] = \
				numpy.zeros((out_size,)).astype('float32')

		self.init_params['skip_W_in2'] = \
				normalized_weight(in2_size, out_size, scale=0.01, ortho=False)

		self.init_params['skip_b_in2'] = \
				numpy.zeros((out_size,)).astype('float32')

	def create_structure(self, theano_params, layer_input_in1, layer_input_in2):
		""" Creates feed-forward layer structure.

		:type theano_params: dict
		:param theano_params: shared Theano variables

		:type layer_input_in1: theano.tensor.var.TensorVariable
		:param layer_input_in1: symbolic matrix that describes the output of the
		first input layer

		:type layer_input_in2: theano.tensor.var.TensorVariable
		:param layer_input_in2: symbolic matrix that describes the output of the
		second input layer

		:rtype: theano.tensor.var.TensorVariable
		:returns: symbolic matrix that describes the output of this layer
		"""

		preact_in1 = tensor.dot(layer_input_in1, theano_params['skip_W_in1']) \
				+ theano_params['skip_b_in1']
		preact_in2 = tensor.dot(layer_input_in2, theano_params['skip_W_in2']) \
				+ theano_params['skip_b_in2']
		return tensor.tanh(preact_in1 + preact_in2)
