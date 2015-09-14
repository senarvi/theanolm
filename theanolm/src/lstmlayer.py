#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor

from matrixfunctions import orthogonal_weight, normalized_weight, get_submatrix


class LSTMLayer(object):
	"""Long Short-Term Memory Layer
	"""

	def __init__(self, options):
		"""Initializes the parameters for an LSTM layer of a recurrent neural
		network.

		:type options: dict
		:param options: a dictionary of training options
		"""

		self.options = options

		# Initialize the parameters.
		self.init_params = OrderedDict()

		nin = self.options['dim_word']
		dim = self.options['dim']

		num_gates = 3

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
		"""Creates LSTM layer structure.

		In mini-batch training the input is 3-dimensional: the first
		dimension is the time step, the second dimension are the sentences,
		and the third dimension is the word projection.

		:type theano_params: dict
		:param theano_params: shared Theano variables

		:type layer_input: theano.tensor.var.TensorVariable
		:param layer_input: x_(t), symbolic 2D or 3D matrix that describes
		                    the output of the previous layer (word
		                    projections of the sentences)
		"""

		if layer_input.ndim != 3:
			raise ValueError("LSTMLayer.create_minibatch_structure() requires 3-dimensional input.")

		num_time_steps = layer_input.shape[0]
		num_sentences = layer_input.shape[1]
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
		init_state = \
				tensor.unbroadcast(
						tensor.alloc(0., num_sentences, self.layer_size), 0)

		outputs, updates = theano.scan(
				self.__create_time_step,
				sequences = sequences,
				outputs_info = [init_state, init_state],
				non_sequences = non_sequences,
				name = 'encoder_time_steps',
				n_steps = num_time_steps,
				profile = self.options['profile'],
				strict = True)
		return outputs[1]

	def create_onestep_structure(self, theano_params, layer_input, init_state):
		"""Creates LSTM layer structure.

		This function is used for creating a text generator. The input is
		2-dimensional: the first dimension is the time step and the second is
		the word projection.

		:type theano_params: dict
		:param theano_params: shared Theano variables

		:type layer_input: theano.tensor.var.TensorVariable
		:param layer_input: x_(t), symbolic 2D or 3D matrix that describes
		                    the output of the previous layer (word
		                    projections of the sentences)
		"""

		if layer_input.ndim != 2:
			raise ValueError("LSTMLayer.create_onestep_structure() requires 2-dimensional input.")
		
		num_time_steps = layer_input.shape[0]
		self.layer_size = theano_params['encoder_U_candidate'].shape[1]

		mask = tensor.alloc(1., num_time_steps, 1)

		# the same __create_time_step() method is used for creating the one time
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

		output = self.__create_time_step(
				mask,
				x_transformed_gates,
				x_transformed_candidate,
				init_state,
				init_state,
				U_gates,
				U_candidate)
		return output[1]

	def __create_time_step(self, mask, x_gates, x_candidate, C_in, h_in, U_gates, U_candidate):
		"""The LSTM step function for theano.scan(). Creates the structure of
		one time step.

		The required affine transformations have already been applied to the
		input prior to creating the loop. The transformed inputs and the mask
		that will be passed to the step function are vectors when processing a
		mini-batch - each value corresponds to the same time step in a different
		sentence.

		:type mask: theano.tensor.var.TensorVariable
		:param mask: masks out time steps after sentence end

		:type x_gates: theano.tensor.var.TensorVariable
		:param x_gates: concatenation of the input x_(t) transformed using the
		                various gate weights and biases

		:type x_candidate: theano.tensor.var.TensorVariable
		:param x_candidate: input x_(t) transformed using the weight W and bias
		                    b for the new candidate state

		:type C_in: theano.tensor.var.TensorVariable
		:param C_in: C_(t-1), state from the previous time step

		:type h_in: theano.tensor.var.TensorVariable
		:param h_in: h_(t-1), output of the previous time step

		:type U_gates: theano.tensor.var.TensorVariable
		:param U_gates: concatenation of the gate weights to be applied to
		                h_(t-1)

		:type U_candidate: theano.tensor.var.TensorVariable
		:param U_candidate: candidate state weight matrix to be applied to
		                    h_(t-1)
		"""

		preact_gates = tensor.dot(h_in, U_gates)
#		preact_gates = tensor.dot(C_in, U_gates)
		preact_gates += x_gates

		# input, forget, and output gates
		i = tensor.nnet.sigmoid(get_submatrix(preact_gates, 0, self.layer_size))
		f = tensor.nnet.sigmoid(get_submatrix(preact_gates, 1, self.layer_size))
		o = tensor.nnet.sigmoid(get_submatrix(preact_gates, 2, self.layer_size))

		preact_candidate = tensor.dot(h_in, U_candidate)
#		preact_candidate = tensor.dot(C_in, U_candidate)
		preact_candidate += x_candidate

		C_candidate = tensor.tanh(preact_candidate)
		C_out = f * C_in + i * C_candidate
		h_out = o * tensor.tanh(C_out)

		# Apply the mask.
		C_out = mask[:,None] * C_out + (1. - mask)[:,None] * C_in
		h_out = mask[:,None] * h_out + (1. - mask)[:,None] * h_in

		return C_out, h_out
