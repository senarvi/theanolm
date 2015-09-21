#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
import numpy

class TextScorer(object):
	"""Text Scoring Using a Neural Network Language Model
	"""

	def __init__(self, network, profile=False):
		"""Creates a Theano function self.costs_function that computes the
		negative log probabilities of given text sequences.

		:type network: RNNLM
		:param network: the neural network object

		:type profile: bool
		:param profile: if set to True, creates a Theano profile object
		"""

		inputs = [network.minibatch_input, network.minibatch_mask]

		# Calculate negative log probability of each word.
		costs = -tensor.log(network.minibatch_probs)
		# Apply mask to the costs matrix.
		costs = costs * network.minibatch_mask
		# Sum costs over time steps to get the negative log probability of each
		# sequence.
		outputs = costs.sum(0)

		self.score_function = \
				theano.function(inputs, outputs, profile=profile)

	def negative_log_probability(self, minibatch_iterator):
		"""Computes the mean negative log probability of mini-batches read using
		the given iterator.

		:type minibatch_iterator: MinibatchIterator
		:param minibatch_iterator: iterator to the input file

		:rtype: float
		:returns: average sequence negative log probability
		"""

		costs = []
		for input_matrix, mask in minibatch_iterator:
			# Append costs of each sequence in the mini-batch. 
			costs.extend(self.score_function(input_matrix, mask))
			if numpy.isnan(numpy.mean(costs)):
				import ipdb; ipdb.set_trace()

		# Return the average sequence cost.
		return numpy.array(costs).mean()
