#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.trainers.modeltrainer import ModelTrainer
import numpy

class NesterovTrainer(ModelTrainer):
	"""Nesterov Momentum Optimization Method

	Normally Nesterov momentum is implemented by first taking a step towards
	the previous update direction, calculating gradient at that position,
	using the gradient to obtain the new update direction, and finally
	updating the parameters. We use an alternative formulation that requires
	the gradient to be computed only at the current parameter values,
	described here:
	https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617

	v_{t} = mu * v_{t-1} - lr * gradient(params_{t-1})
	params_{t} = params_{t-1} + mu * v_{t} - lr * gradient(params_{t-1})
	"""
	
	def __init__(self, network, momentum=0.9, profile=False):
		"""Creates a Nesterov momentum trainer.

		:type network: RNNLM
		:param network: the neural network object

		:type momentum: float
		:param momentum: geometric rate for averaging velocities, i.e. how
		                 much to retain the previous update direction

		:type profile: bool
		:param profile: if set to True, creates a Theano profile object
		"""
		
		self.param_init_values = dict()
		for name, param in network.params.items():
			self.param_init_values[name + '_gradient'] = numpy.zeros_like(param.get_value())
			self.param_init_values[name + '_velocity'] = numpy.zeros_like(param.get_value())
		self._create_params()
		self._momentum = momentum
		self._epsilon = 1e-8
	
		super().__init__(network, profile)
	
	def _get_gradient_updates(self):
		result = []
		for name, gradient_new in zip(self.network.params, self._gradient_wrt_params):
			gradient = self.params[name + '_gradient']
			result.append((gradient, gradient_new))
		return result
	
	def _get_model_updates(self):
		result = []
		for name, param in self.network.params.items():
			gradient = self.params[name + '_gradient']
			velocity = self.params[name + '_velocity']
			standard_update = -(self.learning_rate * gradient)
			velocity_new = (self._momentum * velocity) + standard_update
			param_new = param + (self._momentum * velocity_new) + standard_update
			result.append((velocity, velocity_new))
			result.append((param, param_new))
		return result
