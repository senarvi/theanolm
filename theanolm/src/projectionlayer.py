#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import theano.tensor as tensor

from matrixfunctions import orthogonal_weight

class ProjectionLayer(object):
	"""Projection Layer for Neural Network Language Model
	"""

	def __init__(self, in_size, out_size, options):
		"""Initializes the parameters for the first layer of a neural network
		language model, which creates the word embeddings.

		:type in_size: int
		:param options: dimensionality of the input vectors, i.e. vocabulary
		                size

		:type out_size: int
		:param options: dimensionality of the word projections

		:type options: dict
		:param options: a dictionary of training options
		"""

		self.options = options

		# Initialize the parameters.
		self.param_init_values = OrderedDict()

		self.param_init_values['word_projection'] = \
				orthogonal_weight(in_size, out_size, scale=0.01)

	def create_minibatch_structure(self, model_params, layer_input):
		"""Creates projection layer structure for mini-batch processing.

		Creates the layer structure for 2-dimensional input: the first
		dimension is the time step (index of word in a sequence) and the
		second dimension are the sequences

		:type model_params: dict
		:param model_params: shared Theano variables

		:type layer_input: theano.tensor.var.TensorVariable
		:param layer_input: symbolic 2-dimensional matrix that describes
		                    the input

		:rtype: theano.tensor.var.TensorVariable
		:returns: symbolic 3-dimensional matrix - the first dimension is
		          the time step, the second dimension are the sequences,
		          and the third dimension is the word projection
		"""

		num_time_steps = layer_input.shape[0]
		num_sequences = layer_input.shape[1]

		# Indexing the word_projection matrix with a word ID returns the
		# word_projection_dim dimensional projection. Note that indexing the
		# matrix with a vector of all the word IDs gives a concatenation of
		# those projections.
		projections = model_params['word_projection'][layer_input.flatten()]
		projections = projections.reshape([
				num_time_steps,
				num_sequences,
				self.options['word_projection_dim']])

		# Shift the projections matrix one time step down, setting the first
		# time step to zero projection vectors.
		zero_matrix = tensor.zeros_like(projections)
		self.minibatch_output = tensor.set_subtensor(zero_matrix[1:], projections[:-1])

	def create_onestep_structure(self, model_params, layer_input):
		"""Creates projection layer structure for one-step processing.

		Creates the layer structure for 1-dimensional input. Simply
		indexes the word projection matrix with each word ID.

		:type model_params: dict
		:param model_params: shared Theano variables

		:type layer_input: theano.tensor.var.TensorVariable
		:param layer_input: symbolic vector that describes the word IDs
		                    at the input at this time step (in theory
		                    many sequences could be processed in
		                    parallel).

		:rtype: theano.tensor.var.TensorVariable
		:returns: symbolic 2-dimensional matrix that describes the word
		          projections
		"""

		word_projection_dim = model_params['word_projection'].shape[1]

		# The generation starts with input value -1, which will be translated
		# into zero word projection vector.
		self.onestep_output = tensor.switch(
				layer_input[:,None] < 0,
				tensor.alloc(0.0, 1, word_projection_dim),
				model_params['word_projection'][layer_input])
