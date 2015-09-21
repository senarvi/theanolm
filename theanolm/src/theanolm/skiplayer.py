#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from collections import OrderedDict
import theano.tensor as tensor
from theanolm.matrixfunctions import random_weight

class SkipLayer(object):
	""" Skip-Layer for Neural Network Language Model
	
	A skip-layer combines two inputs by addition (in our case word projections
	with hidden layer outputs).
	"""

	def __init__(self, in1_size, in2_size, out_size):
		"""Initializes the parameters for a skip-layer of a neural network.

		:type in1_size: int
		:param in1_size: number of connections from the first input layer 
	
		:type in2_size: int
		:param in2_size: number of connections from the second input layer
	
		:type out_size: int
		:param out_size: number of output connections
		"""

		# Create the parameters.
		self.param_init_values = OrderedDict()

		self.param_init_values['skip_W_in1'] = \
				random_weight(in1_size, out_size, scale=0.01)

		self.param_init_values['skip_b_in1'] = \
				numpy.zeros((out_size,)).astype('float32')

		self.param_init_values['skip_W_in2'] = \
				random_weight(in2_size, out_size, scale=0.01)

		self.param_init_values['skip_b_in2'] = \
				numpy.zeros((out_size,)).astype('float32')

	def create_structure(self, model_params, layer_input_in1, layer_input_in2):
		""" Creates skip-layer structure.

		Sets self.output to a symbolic matrix that describes the output of this
		layer.

		:type model_params: dict
		:param model_params: shared Theano variables

		:type layer_input_in1: theano.tensor.var.TensorVariable
		:param layer_input_in1: symbolic matrix that describes the output of the
		first input layer

		:type layer_input_in2: theano.tensor.var.TensorVariable
		:param layer_input_in2: symbolic matrix that describes the output of the
		second input layer
		"""

		preact_in1 = tensor.dot(layer_input_in1, model_params['skip_W_in1']) \
				+ model_params['skip_b_in1']
		preact_in2 = tensor.dot(layer_input_in2, model_params['skip_W_in2']) \
				+ model_params['skip_b_in2']
		self.output = tensor.tanh(preact_in1 + preact_in2)
