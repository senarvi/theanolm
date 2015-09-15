#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class TextSampler(object):
	"""Neural network language model sampler

	A Theano function that generates text using a neural network language
	model.
	"""

	def __init__(self, network, options):
		"""Creates the neural network architecture.

		Creates the function self.step_function that uses the state of the
		previous time step and the word ID of the current time step, to compute
		the output distribution. It samples from the output distribution and
		returns the sampled word ID along with the output state of this time
		step.
		 
		:type network: RNNLM
		:param network: the neural network object

		:type options: dict
		:param options: a dictionary of training options
		"""

		self.network = network
		self.options = options
		self.trng = RandomStreams(1234)

		# x describes a sequence of word IDs.
		self.x = tensor.vector('sampler_x', dtype='int64')
		self.x.tag.test_value = \
				numpy.random.randint(0, self.options['vocab_size'], size=4).astype('int64')
		
		# hidden_layer_state describes the state outputs of the previous time
		# step of the hidden layer. GRU has one state output, LSTM has two.
		hidden_layer_state = [tensor.matrix('hidden_layer_state_' + str(i), dtype='float32')
				for i in range(self.network.hidden_layer.num_state_variables)]
		for state_variable in hidden_layer_state:
			state_variable.tag.test_value = numpy.random.rand(
					1,
					self.options['hidden_layer_size']).astype('float32')

		# Create the network structure.
		word_projections = network.projection_layer.create_onestep_structure(
				network.theano_params,
				self.x)
		hidden_layer_outputs = network.hidden_layer.create_onestep_structure(
				network.theano_params,
				word_projections,
				hidden_layer_state)
		skiplayer_output = network.skip_layer.create_structure(
				network.theano_params,
				hidden_layer_outputs[-1],  # The last state is the output.
				word_projections)
		word_probs = network.output_layer.create_onestep_structure(
				network.theano_params,
				skiplayer_output)
		
		# Sample from the output distribution.
		word_ids = self.trng.multinomial(pvals=word_probs).argmax(1)

		# Compile the function.
		print("Building text sampler.")
		inputs = [self.x]
		inputs.extend(hidden_layer_state)
		outputs = [word_ids]
		outputs.extend(hidden_layer_outputs)
		self.step_function = theano.function(
				inputs, outputs, name='text_sampler',
				profile=self.options['profile'])
		print("Done.")

	def generate(self, max_length=30):
		""" Generates a text sequence.

		Calls self.step_function() repeatedly, reading the word output and
		the state output of the hidden layer and passing the hidden layer state
		output to the next time step.
		
		:rtype: list of strs
		:returns: list of the generated words 
		"""

		# -1 indicates the first word of a sequence. We are only generating one
		# sequence at a time.
		word_ids = -1 * numpy.ones(shape=(1,)).astype('int64')
		
		# Construct a list of hidden layer state variables and initialize them
		# to zeros. GRU has only one state that travels through the time steps,
		# LSTM has two.
		hidden_state_shape = (1, self.options['hidden_layer_size'])
		hidden_layer_state = [
				numpy.zeros(shape=hidden_state_shape).astype('float32')
				for _ in range(self.network.hidden_layer.num_state_variables)]
		
		result = []

		for _ in range(max_length):
			step_result = self.step_function(word_ids, *hidden_layer_state)
			word_ids = step_result[0]
			hidden_layer_state = step_result[1:]

			word_id = word_ids[0]
			result.append(word_id)
			if word_id == 0:
				break

		return result
