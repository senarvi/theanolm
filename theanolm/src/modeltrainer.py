#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
import numpy

class ModelTrainer(object):
	"""Superclass for Neural Network Language Model Trainers
	"""

	def __init__(self, network, options):
		"""Creates Theano functions for training a neural network language
		model.

		The subclass constructor is expected to give default values to
		all the required parameters in self.param_init_values first. This
		constructor will then create the corresponding Theano shared
		variables, and two update functions:

		* gradient_update_function: updates the gradient parameters and
		  returns the cost
		* model_update_function: updates model state given the gradients
		  and the learning rate

		:type network: RNNLM
		:param network: the neural network object

		:type options: dict
		:param options: a dictionary of training options
		"""

		self.network = network

		# Calculate negative log probability of each word.
		costs = -tensor.log(self.network.minibatch_probs)
		# Apply mask to the costs matrix.
		costs = costs * self.network.minibatch_mask
		# Sum costs over time steps and take the average over sequences.
		cost = costs.sum(0).mean()
		# Compute the symbolic gradient of each parameter.
		self.gradients = tensor.grad(cost, wrt=list(self.network.params.values()))
		self.gradient_update_function = theano.function(
				[self.network.minibatch_input, self.network.minibatch_mask],
				cost,
				updates=self._get_gradient_updates(),
				profile=options['profile'])

		self.learning_rate = tensor.scalar('learning_rate', dtype='float32')
		self.learning_rate.tag.test_value = numpy.float32('0.001')
		self.model_update_function = theano.function(
				[self.learning_rate],
				[],
				updates=self._get_model_updates(),
				on_unused_input='ignore',
				profile=options['profile'])

	def _create_params(self):
		"""Creates Theano shared variables from the initial parameter values.
		"""

		self.params = {name: theano.shared(value, name)
				for name, value in self.param_init_values.items()}

class SGDTrainer(ModelTrainer):
	"""Stochastic Gradient Descent Trainer for a Neural Network Language Model
	"""

	def __init__(self, network, options):
		self.param_init_values = {name + '_gradient': param.get_value() * 0.0
				for name, param in network.params.items()}
		self._create_params()
		self.gradient_params = [self.params[name + '_gradient']
				for name in network.params]

		super().__init__(network, options)

	def _get_gradient_updates(self):
		return [(param, gradient)
				for param, gradient in zip(self.gradient_params, self.gradients)]

	def _get_model_updates(self):
		return [(param, param - self.learning_rate * gradient)
				for param, gradient in zip(self.network.params.values(), gradient_params)]

class AdamTrainer(ModelTrainer):
	"""Adam Trainer for a Neural Network Language Model

	D.P. Kingma, J. Ba
	Adam: A Method for Stochastic Optimization
	The International Conference on Learning Representations (ICLR), San Diego, 2015
	"""

	def __init__(self, network, options):
		self.param_init_values = dict()
		for name, param in network.params.items():
			self.param_init_values[name + '_gradient'] = param.get_value() * 0.0
			self.param_init_values[name + '_adam_m'] = param.get_value() * 0.0
			self.param_init_values[name + '_adam_v'] = param.get_value() * 0.0
		self.param_init_values['adam_timestep'] = numpy.float32(0.0)
		self._create_params()
		self.gradient_params = [self.params[name + '_gradient']
				for name in network.params]
		self.timestep = self.params['adam_timestep']

		super().__init__(network, options)

	def _get_gradient_updates(self):
		return [(param, gradient)
				for param, gradient in zip(self.gradient_params, self.gradients)]

	def _get_model_updates(self):
		b1 = 0.9
		b2 = 0.999
		e = 1e-8

		timestep_new = self.timestep + 1.0
		alpha = self.learning_rate * tensor.sqrt(1.0 - b2**timestep_new) \
				/ (1.0 - b1**timestep_new)

		result = []
		for name, param, gradient in zip(self.network.params, self.network.params.values(), self.gradient_params):
			m = self.params[name + '_adam_m']
			v = self.params[name + '_adam_v']
			m_new = (b1 * m) + ((1.0 - b1) * gradient)
			v_new = (b2 * v) + ((1.0 - b2) * tensor.sqr(gradient))
			param_new = param - (alpha * m_new / (tensor.sqrt(v_new) + e))
			result.append((m, m_new))
			result.append((v, v_new))
			result.append((param, param_new))
		result.append((self.timestep, timestep_new))
		return result
