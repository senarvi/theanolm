#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor

from matrixfunctions import orthogonal_weight, normalized_weight, get_submatrix


class GRULayer(object):
	"""Gated Recurrent Unit Layer
	"""

	def __init__(self, options):
		"""Initializes the parameters for a GRU layer of a recurrent neural
		network.

		:type options: dict
		:param options: a dictionary of training options
		"""

		self.options = options

		# Initialize the parameters.
		self.init_params = OrderedDict()

		nin = self.options['word_projection_dim']
		dim = self.options['hidden_layer_size']
		
		num_gates = 2

		# concatenation of the input weights for each gate
		self.init_params['encoder_W_gates'] = \
				numpy.concatenate(
						[normalized_weight(nin, dim) for _ in range(num_gates)],
						axis=1)

		# concatenation of the previous step output weights for each gate
		self.init_params['encoder_U_gates'] = \
				numpy.concatenate(
						[orthogonal_weight(dim) for _ in range(num_gates)],
						axis=1)

		# concatenation of the biases for each gate
		self.init_params['encoder_b_gates'] = \
				numpy.zeros((num_gates * dim,)).astype('float32')
		
		# input weight for the candidate state
		self.init_params['encoder_W_candidate'] = \
				normalized_weight(nin, dim)

		# previous step output weight for the candidate state
		self.init_params['encoder_U_candidate'] = \
				orthogonal_weight(dim)
		
		# bias for the candidate state
		self.init_params['encoder_b_candidate'] = \
				numpy.zeros((dim,)).astype('float32') 

	def create_minibatch_structure(self, theano_params, layer_input, mask):
		"""Creates GRU layer structure.

		In mini-batch training the input is 3-dimensional: the first
		dimension is the time step, the second dimension are the sequences,
		and the third dimension is the word projection.

		:type theano_params: dict
		:param theano_params: shared Theano variables

		:type layer_input: theano.tensor.var.TensorVariable
		:param layer_input: x_(t), symbolic 3-dimensional matrix that
		                    describes the output of the previous layer (word
		                    projections of the sequences)
		"""

		if layer_input.ndim != 3:
			raise ValueError("GRULayer.create_minibatch_structure() requires 3-dimensional input.")

		num_time_steps = layer_input.shape[0]
		num_sequences = layer_input.shape[1]
		self.layer_size = theano_params['encoder_U_candidate'].shape[1]

		# Before looping, the weights and biases are applied to the input,
		# which does not depend on the time step.
		x_transformed_gates = \
				tensor.dot(layer_input, theano_params['encoder_W_gates']) \
				+ theano_params['encoder_b_gates']
		x_transformed_candidate = \
				tensor.dot(layer_input, theano_params['encoder_W_candidate']) \
				+ theano_params['encoder_b_candidate']
		
		# The weights and biases for the previous step output. These will
		# be applied inside the loop.
		U_gates = theano_params['encoder_U_gates']
		U_candidate = theano_params['encoder_U_candidate']

		sequences = [mask, x_transformed_gates, x_transformed_candidate]
		non_sequences = [U_gates, U_candidate]
		init_state = tensor.unbroadcast(
				tensor.alloc(0.0, num_sequences, self.layer_size), 0)

		outputs, updates = theano.scan(
				self.__create_time_step,
				sequences = sequences,
				outputs_info = [init_state],
				non_sequences = non_sequences,
				name = 'encoder_time_steps',
				n_steps = num_time_steps,
				profile = self.options['profile'],
				strict = True)
		return outputs

	def create_onestep_structure(self, theano_params, layer_input, init_state):
		"""Creates GRU layer structure.

		This function is used for creating a text generator. The input is
		2-dimensional: the first dimension is the sequence and the second is
		the word projection.

		:type theano_params: dict
		:param theano_params: shared Theano variables

		:type layer_input: theano.tensor.var.TensorVariable
		:param layer_input: x_(t), symbolic 2-dimensional matrix that
		                    describes the output of the previous layer (word
		                    projections of the sequences)
		"""

		if layer_input.ndim != 2:
			raise ValueError("GRULayer.create_onestep_structure() requires 2-dimensional input.")

		num_sequences = layer_input.shape[0]
		self.layer_size = theano_params['encoder_U_candidate'].shape[1]

		mask = tensor.alloc(1.0, num_sequences, 1)

		# The same __create_time_step() method is used for creating the one time
		# step, so we have to apply the weights and biases first.
		x_transformed_gates = \
				tensor.dot(layer_input, theano_params['encoder_W_gates']) \
				+ theano_params['encoder_b_gates']
		x_transformed_candidate = \
				tensor.dot(layer_input, theano_params['encoder_W_candidate']) \
				+ theano_params['encoder_b_candidate']
		
		# The weights and biases for the previous step output. These will
		# be applied inside __create_time_step().
		U_gates = theano_params['encoder_U_gates']
		U_candidate = theano_params['encoder_U_candidate']

		outputs = self.__create_time_step(
				mask,
				x_transformed_gates,
				x_transformed_candidate,
				init_state,
				U_gates,
				U_candidate)
		return outputs

	def __create_time_step(self, mask, x_gates, x_candidate, h_in, U_gates, U_candidate):
		"""The GRU step function for theano.scan(). Creates the structure of one
		time step.

		The required affine transformations have already been applied to the
		input prior to creating the loop. The transformed inputs and the mask
		that will be passed to the step function are vectors when processing a
		mini-batch - each value corresponds to the same time step in a different
		sequence.

		:type mask: theano.tensor.var.TensorVariable
		:param mask: masks out time steps after sequence end

		:type x_gates: theano.tensor.var.TensorVariable
		:param x_gates: concatenation of the input x_(t) transformed using the various
		                gate weights and biases

		:type x_candidate: theano.tensor.var.TensorVariable
		:param x_candidate: input x_(t) transformed using the weight W and bias b
		                    for the new candidate state

		:type h_in: theano.tensor.var.TensorVariable
		:param h_in: h_(t-1), hidden state output of the previous time step

		:type U_gates: theano.tensor.var.TensorVariable
		:param U_gates: concatenation of the gate weights to be applied to h_(t-1)

		:type U_candidate: theano.tensor.var.TensorVariable
		:param U_candidate: candidate state weight matrix to be applied to h_(t-1)
		"""

		# pre-activation of the gates
		preact_gates = tensor.dot(h_in, U_gates)
		preact_gates += x_gates

		# reset and update gates
		r = tensor.nnet.sigmoid(get_submatrix(preact_gates, 0, self.layer_size))
		u = tensor.nnet.sigmoid(get_submatrix(preact_gates, 1, self.layer_size))

		# pre-activation of the candidate state
		preact_candidate = tensor.dot(h_in, U_candidate)
		preact_candidate *= r
		preact_candidate += x_candidate

		# hidden state output
		h_candidate = tensor.tanh(preact_candidate)
		h_out = (1.0 - u) * h_in + u * h_candidate

		# Apply the mask.
		h_out = mask[:,None] * h_out + (1.0 - mask)[:,None] * h_in

		return h_out
