#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theanolm.trainers.modeltrainer import ModelTrainer
import theano.tensor as tensor
import numpy

class AdamTrainer(ModelTrainer):
	"""Adam Optimization Method

	D.P. Kingma, J. Ba
	Adam: A Method for Stochastic Optimization
	The International Conference on Learning Representations (ICLR), San Diego, 2015
	"""

	def __init__(self, network, profile):
		"""Creates an Adam trainer.

		:type network: RNNLM
		:param network: the neural network object

		:type profile: bool
		:param profile: if set to True, creates a Theano profile object
		"""
		
		self.param_init_values = dict()
		for name, param in network.params.items():
			self.param_init_values[name + '_gradient'] = numpy.zeros_like(param.get_value())
			self.param_init_values[name + '_mean_gradient'] = numpy.zeros_like(param.get_value())
			self.param_init_values[name + '_mean_sqr_gradient'] = numpy.zeros_like(param.get_value())
		self.param_init_values['adam_timestep'] = numpy.float32(0.0)
		self._create_params()
		self._timestep = self.params['adam_timestep']
		self._gamma_m = 0.9  # geometric rate for averaging gradients (decay rate)
		self._gamma_ms = 0.999  # geometric rate for averaging squared gradients (decay rate)
		self._epsilon = 1e-8

		super().__init__(network, profile)

	def _get_gradient_updates(self):
		result = []
		for name, gradient_new in zip(self.network.params, self._gradient_wrt_params):
			gradient = self.params[name + '_gradient']
			m_gradient = self.params[name + '_mean_gradient']
			ms_gradient = self.params[name + '_mean_sqr_gradient']
			m_gradient_new = (self._gamma_m * m_gradient) + ((1.0 - self._gamma_m) * gradient)
			ms_gradient_new = (self._gamma_ms * ms_gradient) + ((1.0 - self._gamma_ms) * tensor.sqr(gradient))
			result.append((gradient, gradient_new))
			result.append((m_gradient, m_gradient_new))
			result.append((ms_gradient, ms_gradient_new))
		return result

	def _get_model_updates(self):
		timestep_new = self._timestep + 1.0
		alpha = self.learning_rate * tensor.sqrt(1.0 - (self._gamma_ms ** timestep_new)) \
				/ (1.0 - (self._gamma_m ** timestep_new))

		result = []
		for name, param in self.network.params.items():
			m_gradient = self.params[name + '_mean_gradient']
			ms_gradient = self.params[name + '_mean_sqr_gradient']
			rms_gradient = tensor.sqrt(ms_gradient) + self._epsilon
			param_new = param - (alpha * m_gradient / rms_gradient)
			result.append((param, param_new))
		result.append((self._timestep, timestep_new))
		return result
