#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import warnings
import os
import numpy
import theano
import theano.tensor as tensor

from projectionlayer import ProjectionLayer
from lstmlayer import LSTMLayer
from sslstmlayer import SSLSTMLayer
from grulayer import GRULayer
from skiplayer import SkipLayer
from outputlayer import OutputLayer
from matrixfunctions import test_value

class RNNLM(object):
	"""Recursive Neural Network Language Model

	A recursive neural network language model implemented using Theano.
	Supports LSTM and GRU architectures.
	"""

	def __init__(self, dictionary, options):
		"""Initializes the neural network parameters for all layers, and
		creates Theano shared variables from them.

		:type dictionary: Dictionary
		:param dictionary: mapping between word IDs and word classes

		:type options: dict
		:param options: a dictionary of training options
		"""

		self.dictionary = dictionary
		self.options = options

		# This class stores the training error history too, because it's saved
		# in the same .npz file.
		self.error_history = []

		# Create the layers.
		
		self.projection_layer = ProjectionLayer(
				dictionary.num_classes(),
				self.options['word_projection_dim'],
				self.options)
		
		if self.options['hidden_layer_type'] == 'lstm':
			self.hidden_layer = LSTMLayer(
					self.options['word_projection_dim'],
					self.options['hidden_layer_size'],
					self.options)
		elif self.options['hidden_layer_type'] == 'ss-lstm':
			self.hidden_layer = SSLSTMLayer(
					self.options['word_projection_dim'],
					self.options['hidden_layer_size'],
					self.options)
		elif self.options['hidden_layer_type'] == 'gru':
			self.hidden_layer = GRULayer(
					self.options['word_projection_dim'],
					self.options['hidden_layer_size'],
					self.options)
		else:
			raise ValueError("Invalid hidden layer type: " + self.options['hidden_layer_type'])
		
		self.skip_layer = SkipLayer(
				self.options['hidden_layer_size'],
				self.options['word_projection_dim'],
				self.options['word_projection_dim'])
		
		self.output_layer = OutputLayer(
				options['word_projection_dim'],
				dictionary.num_classes())

		# Initialize the parameters.
		self.init_params = OrderedDict()
		self.init_params.update(self.projection_layer.init_params)
		self.init_params.update(self.hidden_layer.init_params)
		self.init_params.update(self.skip_layer.init_params)
		self.init_params.update(self.output_layer.init_params)

		# Reload the parameters from disk if requested.
		if self.options['reload_state'] and os.path.exists(self.options['model_path']):
			self.__load_params()

		# Create Theano shared variables.
		self.theano_params = OrderedDict()
		for name, value in self.init_params.items():
			self.theano_params[name] = theano.shared(value, name)
		
		self.create_minibatch_structure()
		self.create_onestep_structure()

	def create_minibatch_structure(self):
		"""Creates the network structure for mini-batch processing.
		
		Sets self.minibatch_output to a symbolic matrix, the same shape as
		self.minibatch_input, containing the output word probabilities, and
		self.minibatch_costs containing the negative log probabilities. 
		"""

		# minibatch_input describes the input matrix containing
		# [ number of time steps * number of sequences ] word IDs.
		self.minibatch_input = tensor.matrix('minibatch_input', dtype='int64')
		self.minibatch_input.tag.test_value = test_value(
				size=(16, 4),
				max_value=self.dictionary.num_classes())
		
		# mask is used to mask out the rest of the input matrix, when a sequence
		# is shorter than the maximum sequence length.
		self.minibatch_mask = tensor.matrix('minibatch_mask', dtype='float32')
		self.minibatch_mask.tag.test_value = test_value(
				size=(16, 4),
				max_value=1.0)

		self.projection_layer.create_minibatch_structure(
				self.theano_params,
				self.minibatch_input)
		self.hidden_layer.create_minibatch_structure(
				self.theano_params,
				self.projection_layer.minibatch_output,
				mask=self.minibatch_mask)
		self.skip_layer.create_structure(
				self.theano_params,
				self.hidden_layer.minibatch_output,
				self.projection_layer.minibatch_output)
		self.output_layer.create_minibatch_structure(
				self.theano_params,
				self.skip_layer.output)
		
		self.minibatch_output = self.output_layer.minibatch_output
		
		# Input word IDs + the index times vocabulary size can be used to index
		# a flattened output matrix to read the probabilities of the input
		# words.
		input_flat = self.minibatch_input.flatten()
		flat_output_indices = \
				tensor.arange(input_flat.shape[0]) * self.dictionary.num_classes() \
				+ input_flat
		input_word_probs = self.minibatch_output.flatten()[flat_output_indices]
		input_word_probs = input_word_probs.reshape(
				[self.minibatch_input.shape[0], self.minibatch_input.shape[1]])
		self.minibatch_probs = input_word_probs

	def create_onestep_structure(self):
		"""Creates the network structure for one-step processing.
		"""

		# onestep_input describes the input vector containing as many word IDs
		# as there are sequences (only one for the text sampler).
		self.onestep_input = tensor.vector('onestep_input', dtype='int64')
		self.onestep_input.tag.test_value = test_value(
				size=4,
				max_value=self.dictionary.num_classes())
		
		# onestep_state describes the state outputs of the previous time step
		# of the hidden layer. GRU has one state output, LSTM has two.
		self.onestep_state = [tensor.matrix('onestep_state_' + str(i), dtype='float32')
				for i in range(self.hidden_layer.num_state_variables)]
		for state_variable in self.onestep_state:
			state_variable.tag.test_value = test_value(
					size=(1, self.options['hidden_layer_size']),
					max_value=1.0)

		self.projection_layer.create_onestep_structure(
				self.theano_params,
				self.onestep_input)
		self.hidden_layer.create_onestep_structure(
				self.theano_params,
				self.projection_layer.onestep_output,
				self.onestep_state)
		# The last state output from the hidden layer is the hidden state to be
		# passed on the the next layer.
		hidden_state_output = self.hidden_layer.onestep_outputs[-1]
		self.skip_layer.create_structure(
				self.theano_params,
				hidden_state_output,
				self.projection_layer.onestep_output)
		self.output_layer.create_onestep_structure(
				self.theano_params,
				self.skip_layer.output)
		
		self.onestep_output = self.output_layer.onestep_output

	def __load_params(self):
		"""Loads the neural network parameters from disk.
		
		This function should be called only in the constructor, before the
		Theano shared variables are created.
		"""

		path = self.options['model_path']

		# Reload the parameters.
		data = numpy.load(path)
		num_updated = 0
		for name in self.init_params:
			if name not in data:
				warnings.warn('The parameter %s was not found from the archive.' % name)
				continue
			self.init_params[name] = data[name]
			num_updated += 1
		print("Read %d parameter values from %s." % (num_updated, path))

		# Reload the error history.
		if 'error_history' not in data:
			warnings.warn('Training error history was not found from the archive.' % name)
		else:
			saved_error_history = data['error_history'].tolist()
			# If the error history was empty when the state was saved,
			# ndarray.tolist() will return None.
			if not saved_error_history is None:
				self.error_history = saved_error_history

		self.print_error_history()

	def save_params(self, params=None):
		"""Saves the neural network parameters to disk.

		:type params: dict
		:param params: if set to other than None, save these values, instead of the
				  current values from the Theano shared variables
		"""

		path = self.options['model_path']
		if params is None:
			params = self.get_param_values()

		numpy.savez(path, error_history=self.error_history, **params)
		print("Saved %d parameter values to %s." % (len(params), path))

	def print_error_history(self):
		"""Prints the current error history.
		
		We're using numpy for prettier formatting.
		"""

		print("Training error history:")
		print(numpy.asarray(self.error_history))

	def get_param_values(self):
		"""Pulls parameter values from Theano shared variables.

		:rtype: dict
		:returns: a dictionary of the parameter values
		"""

		result = OrderedDict()
		for name, param in self.theano_params.items():
			result[name] = param.get_value()
		return result

	def set_param_values(self, x):
		"""Sets the values of Theano shared variables.

		:type x: dict
		:param x: a dictionary of the new parameter values
		"""

		for name, value in x.items():
			self.theano_params[name].set_value(value)
