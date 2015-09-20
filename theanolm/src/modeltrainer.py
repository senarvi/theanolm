#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import theano
import theano.tensor as tensor
import numpy

from exceptions import InputError, NumberError

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

		self.update_number = 0
		self._cost_history = []

	def _create_params(self):
		"""Creates Theano shared variables from the initial parameter values.
		"""

		self.params = {name: theano.shared(value, name)
				for name, value in self.param_init_values.items()}

	def load_params(self, path):
		"""Loads the training state from disk.

		:type path: str
		:param path: filesystem path where to read the state from
		"""

		data = numpy.load(path)
		num_updated = 0
		for name in self.params:
			if not name in data:
				raise InputError("Parameter %s was not found from %s." % (name, path))
			self.params[name].set_value(data[name])
			num_updated += 1
		print("Read %d parameter values from %s." % (num_updated, path))

		if not 'cost_history' in data:
			raise InputError("Validation set cost history was not found from %s." % (name, path))
		saved_cost_history = data['cost_history'].tolist()
		# If the error history was empty when the state was saved,
		# ndarray.tolist() will return None.
		if saved_cost_history is None:
			self._cost_history = []
		else:
			self._cost_history = saved_cost_history
		self.print_cost_history()

		if not 'update_number' in data:
			raise InputError("Current update number was not found from %s." % (name, path))
		self.update_number = data['update_number']
		print("Previous training has been stopped after update %d." % self.update_number)

	def save_params(self, path):
		"""Saves the current training state to disk.

		:type path: str
		:param path: filesystem path where to save the parameters to
		"""

		numpy.savez(path, update_number=self.update_number, cost_history=self._cost_history, **params)
		print("Saved %d parameter values and error history to %s." % (len(params), path))

	def print_cost_history(self):
		"""Prints the current error history.
		
		We're using numpy for prettier formatting.
		"""

		print("Validation set cost history during training:")
		print(numpy.asarray(self._cost_history))

	def update_minibatch(self, x, mask, learning_rate, verbose=False):
		"""Optimizes the neural network parameters using the given inputs
		and learning rate.

		:type x: numpy.matrixlib.defmatrix.matrix
		:param x: a 2-dimensional matrix indexed by time step and sequence
		          that contains word IDs

		:type mask: numpy.matrixlib.defmatrix.matrix
		:param mask: a matrix the same shape as x that contains 0.0 where x
		             contains input and 1.0 after sequence end

		:type learning_rate: float
		:param learning_rate: (initial) learning rate for the optimization

		:type verbose: bool
		:param verbos: if set to True, prints update cost and duration
		"""

		self.update_number += 1

		update_start_time = time.time()
		cost = self.gradient_update_function(x, mask)
		if numpy.isnan(cost):
			raise NumberError("Update %d cost has NaN value." % self.update_number)
		if numpy.isinf(cost):
			raise NumberError("Update %d cost has infinite value." % self.update_number)
		self.model_update_function(learning_rate)
		update_duration = time.time() - update_start_time

		if verbose:
			print("Update %d -- mini-batch cost = %f, duration = %f seconds" % (self.update_number, cost, update_duration))

	def append_validation_cost(self, validation_cost):
		"""Adds the validation set cost to the cost history.

		:type validation_cost: float
		:param validation_cost: the new validation set cost to be added to the history
		"""

		self._cost_history.append(validation_cost)
		self.print_cost_history()

	def validations_since_min_cost(self):
		"""Returns the number of times the validation set cost has been computed
		since the minimum cost was obtained.

		:rtype: int
		:returns: number of validations since the minimum cost (0 means the last
		          validation is the best so far)
		"""

		if len(self._cost_history) == 0:
			raise RuntimeError("ModelTrainer.validations_since_min_cost() called with empty cost history.")
		else:
			# Reverse the order of self._cost_history to find the last element
			# with the minimum value (in case there are several elements with the
			# same value.
			return numpy.argmin(self._cost_history[::-1])


class SGDTrainer(ModelTrainer):
	"""Stochastic Gradient Descent Trainer for a Neural Network Language Model
	"""

	def __init__(self, network, options):
		self.param_init_values = {name + '_gradient': param.get_value() * 0.0
				for name, param in network.params.items()}
		self._create_params()
		self._gradient_params = [self.params[name + '_gradient']
				for name in network.params]

		super().__init__(network, options)

	def _get_gradient_updates(self):
		return [(param, gradient)
				for param, gradient in zip(self._gradient_params, self.gradients)]

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
		self._gradient_params = [self.params[name + '_gradient']
				for name in network.params]
		self._timestep = self.params['adam_timestep']

		super().__init__(network, options)

	def _get_gradient_updates(self):
		return [(param, gradient)
				for param, gradient in zip(self._gradient_params, self.gradients)]

	def _get_model_updates(self):
		b1 = 0.9
		b2 = 0.999
		e = 1e-8

		timestep_new = self._timestep + 1.0
		alpha = self.learning_rate * tensor.sqrt(1.0 - b2**timestep_new) \
				/ (1.0 - b1**timestep_new)

		result = []
		for name, param, gradient in zip(self.network.params, self.network.params.values(), self._gradient_params):
			m = self.params[name + '_adam_m']
			v = self.params[name + '_adam_v']
			m_new = (b1 * m) + ((1.0 - b1) * gradient)
			v_new = (b2 * v) + ((1.0 - b2) * tensor.sqr(gradient))
			param_new = param - (alpha * m_new / (tensor.sqrt(v_new) + e))
			result.append((m, m_new))
			result.append((v, v_new))
			result.append((param, param_new))
		result.append((self._timestep, timestep_new))
		return result
