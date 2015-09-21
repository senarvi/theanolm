#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import theano
import theano.tensor as tensor
from theanolm.exceptions import IncompatibleStateError
from theanolm.projectionlayer import ProjectionLayer
from theanolm.lstmlayer import LSTMLayer
from theanolm.sslstmlayer import SSLSTMLayer
from theanolm.grulayer import GRULayer
from theanolm.skiplayer import SkipLayer
from theanolm.outputlayer import OutputLayer
from theanolm.matrixfunctions import test_value

class RNNLM(object):
	"""Recursive Neural Network Language Model

	A recursive neural network language model implemented using Theano.
	Supports LSTM and GRU architectures.
	"""

	def __init__(self, dictionary, word_projection_dim, hidden_layer_type,
	             hidden_layer_size, profile=False):
		"""Initializes the neural network parameters for all layers, and
		creates Theano shared variables from them.

		:type dictionary: Dictionary
		:param dictionary: mapping between word IDs and word classes

		:type word_projection_dim: int
		:param word_projection_dim: dimensionality of the word projections

		:type hidden_layer_type: str
		:param hidden_layer_type: name of the units used in the hidden layer

		:type hidden_layer_size: int
		:param hidden_layer_size: number of units in the hidden layer

		:type profile: bool
		:param profile: if set to True, creates a Theano profile object
		"""

		self.dictionary = dictionary
		self.word_projection_dim = word_projection_dim
		self.hidden_layer_type = hidden_layer_type
		self.hidden_layer_size = hidden_layer_size

		# Create the layers.
		self.projection_layer = ProjectionLayer(
				dictionary.num_classes(),
				self.word_projection_dim)
		if self.hidden_layer_type == 'lstm':
			self.hidden_layer = LSTMLayer(
					self.word_projection_dim,
					self.hidden_layer_size,
					profile)
		elif self.hidden_layer_type == 'ss-lstm':
			self.hidden_layer = SSLSTMLayer(
					self.word_projection_dim,
					self.hidden_layer_size,
					profile)
		elif self.hidden_layer_type == 'gru':
			self.hidden_layer = GRULayer(
					self.word_projection_dim,
					self.hidden_layer_size,
					profile)
		else:
			raise ValueError("Invalid hidden layer type: " + self.hidden_layer_type)
		self.skip_layer = SkipLayer(
				self.hidden_layer_size,
				self.word_projection_dim,
				self.word_projection_dim)
		self.output_layer = OutputLayer(
				self.word_projection_dim,
				dictionary.num_classes())

		# Create initial parameter values.
		self.param_init_values = OrderedDict()
		self.param_init_values.update(self.projection_layer.param_init_values)
		self.param_init_values.update(self.hidden_layer.param_init_values)
		self.param_init_values.update(self.skip_layer.param_init_values)
		self.param_init_values.update(self.output_layer.param_init_values)

		# Create Theano shared variables.
		self.params = {name: theano.shared(value, name)
				for name, value in self.param_init_values.items()}
		
		self._create_minibatch_structure()
		self._create_onestep_structure()

	def _create_minibatch_structure(self):
		"""Creates the network structure for mini-batch processing.
		
		Sets self.minibatch_output to a symbolic matrix, the same shape as
		self.minibatch_input, containing the output word probabilities, and
		self.minibatch_costs containing the negative log probabilities. 
		"""

		# minibatch_input describes the input matrix containing
		# [ number of time steps * number of sequences ] word IDs.
		self.minibatch_input = tensor.matrix('minibatch_input', dtype='int64')
		self.minibatch_input.tag.test_value = test_value(
				size=(100, 16),
				max_value=self.dictionary.num_classes())
		
		# mask is used to mask out the rest of the input matrix, when a sequence
		# is shorter than the maximum sequence length.
		self.minibatch_mask = tensor.matrix('minibatch_mask', dtype='float32')
		self.minibatch_mask.tag.test_value = test_value(
				size=(100, 16),
				max_value=1.0)

		self.projection_layer.create_minibatch_structure(
				self.params,
				self.minibatch_input)
		self.hidden_layer.create_minibatch_structure(
				self.params,
				self.projection_layer.minibatch_output,
				mask=self.minibatch_mask)
		self.skip_layer.create_structure(
				self.params,
				self.hidden_layer.minibatch_output,
				self.projection_layer.minibatch_output)
		self.output_layer.create_minibatch_structure(
				self.params,
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

	def _create_onestep_structure(self):
		"""Creates the network structure for one-step processing.
		"""

		# onestep_input describes the input vector containing as many word IDs
		# as there are sequences (only one for the text sampler).
		self.onestep_input = tensor.vector('onestep_input', dtype='int64')
		self.onestep_input.tag.test_value = test_value(
				size=16,
				max_value=self.dictionary.num_classes())
		
		# onestep_state describes the state outputs of the previous time step
		# of the hidden layer. GRU has one state output, LSTM has two.
		self.onestep_state = [tensor.matrix('onestep_state_' + str(i), dtype='float32')
				for i in range(self.hidden_layer.num_state_variables)]
		for state_variable in self.onestep_state:
			state_variable.tag.test_value = test_value(
					size=(1, self.hidden_layer_size),
					max_value=1.0)

		self.projection_layer.create_onestep_structure(
				self.params,
				self.onestep_input)
		self.hidden_layer.create_onestep_structure(
				self.params,
				self.projection_layer.onestep_output,
				self.onestep_state)
		# The last state output from the hidden layer is the hidden state to be
		# passed on the the next layer.
		hidden_state_output = self.hidden_layer.onestep_outputs[-1]
		self.skip_layer.create_structure(
				self.params,
				hidden_state_output,
				self.projection_layer.onestep_output)
		self.output_layer.create_onestep_structure(
				self.params,
				self.skip_layer.output)
		
		self.onestep_output = self.output_layer.onestep_output

	def get_state(self):
		"""Pulls parameter values from Theano shared variables.

		:rtype: dict
		:returns: a dictionary of the parameter values
		"""

		result = OrderedDict()
		for name, param in self.params.items():
			result[name] = param.get_value()
		result['rnnlm_word_projection_dim'] = self.word_projection_dim
		result['rnnlm_hidden_layer_type'] = self.hidden_layer_type
		result['rnnlm_hidden_layer_size'] = self.hidden_layer_size
		return result

	def set_state(self, state):
		"""Sets the values of Theano shared variables.
		
		Requires that ``state`` contains values for all the neural network
		parameters.

		:type state: dict
		:param state: dictionary of neural network parameters
		"""

		for name, param in self.params.items():
			if not name in state:
				raise IncompatibleStateError("Parameter %s is missing from neural network state." % name)
			param.set_value(state[name])
		if state['rnnlm_word_projection_dim'] != self.word_projection_dim:
			raise IncompatibleStateError("Attempting to restore incompatible state with word_projection_dim=%d, while this neural network has word_projection_dim=%d." % (state['rnnlm_word_projection_dim'], self.word_projection_dim))
		if state['rnnlm_hidden_layer_type'] != self.hidden_layer_type:
			raise IncompatibleStateError("Attempting to restore incompatible state with hidden_layer_type=%s, while this neural network has hidden_layer_type=%s." % (state['rnnlm_hidden_layer_type'], self.hidden_layer_type))
		if state['rnnlm_hidden_layer_size'] != self.hidden_layer_size:
			raise IncompatibleStateError("Attempting to restore incompatible state with hidden_layer_size=%d, while this neural network has hidden_layer_size=%d." % (state['rnnlm_hidden_layer_size'], self.hidden_layer_size))
