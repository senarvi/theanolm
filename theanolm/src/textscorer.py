#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
import numpy

class TextScorer(object):
	"""Text Scoring Using a Neural Network Language Model
	"""

	def __init__(self, network, options):
		"""Creates a Theano function self.score_function that computes the
		negative log probability of a text segment.

		:type network: RNNLM
		:param network: the neural network object

		:type options: dict
		:param options: a dictionary of training options
		"""

		self.network = network
		self.options = options

		input_matrix = self.network.minibatch_input
		input_flat = input_matrix.flatten()

		# Input word IDs + the index times vocabulary size can be used to index
		# a flattened output matrix.
		flat_output_indices = \
				tensor.arange(input_flat.shape[0]) * self.options['vocab_size'] \
				+ input_flat

		# Calculate negative log probability of each word.
		word_probs = self.network.minibatch_output
		cost = -tensor.log(word_probs.flatten()[flat_output_indices])
		cost = cost.reshape([input_matrix.shape[0], input_matrix.shape[1]])

		# Apply mask to the cost matrix.
		mask = self.network.minibatch_mask
		cost = (cost * mask).sum(0)

		inputs = [input_matrix, mask]
		self.score_function = theano.function(inputs, cost, profile=self.options['profile'])

	def negative_log_probability(self, minibatch_iterator):
		word_costs = []

		for input_matrix, mask in minibatch_iterator:
			#pprobs = self.score_function(input_matrix, mask)
			#for pp in pprobs:
			#	word_costs.append(pp)
			word_costs.extend(self.score_function(input_matrix, mask))

			if numpy.isnan(numpy.mean(word_costs)):
				import ipdb; ipdb.set_trace()

		return numpy.array(word_costs).mean()
