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

    def __init__(self, network, training_options, profile):
        """Creates an Adam trainer.

        :type network: Network
        :param network: the neural network object

        :type training_options: dict
        :param training_options: a dictionary of training options

        :type profile: bool
        :param profile: if set to True, creates a Theano profile object
        """

        self.param_init_values = dict()
        for name, param in network.params.items():
            self.param_init_values[name + '.gradient'] = numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_gradient'] = numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_sqr_gradient'] = numpy.zeros_like(param.get_value())
        self.param_init_values['adam.timestep'] = numpy.float32(0.0)
        self._create_params()

        # geometric rate for averaging gradients
        if not 'gradient_decay_rate' in training_options:
            raise ValueError("Gradient decay rate is not given in training options.")
        self._gamma_m = training_options['gradient_decay_rate']

        # geometric rate for averaging squared gradients
        if not 'sqr_gradient_decay_rate' in training_options:
            raise ValueError("Squared gradient decay rate is not given in training options.")
        self._gamma_ms = training_options['sqr_gradient_decay_rate']

        # numerical stability / smoothing term to prevent divide-by-zero
        if not 'epsilon' in training_options:
            raise ValueError("Epsilon is not given in training options.")
        self._epsilon = training_options['epsilon']

        # momentum
        if not 'momentum' in training_options:
            raise ValueError("Momentum is not given in training options.")
        self._momentum = training_options['momentum']

        super().__init__(network, training_options, profile)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params, self._gradient_wrt_params):
            gradient = self.params[name + '.gradient']
            m_gradient = self.params[name + '.mean_gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            m_gradient_new = (self._gamma_m * m_gradient) + ((1.0 - self._gamma_m) * gradient)
            ms_gradient_new = (self._gamma_ms * ms_gradient) + ((1.0 - self._gamma_ms) * tensor.sqr(gradient))
            result.append((gradient, gradient_new))
            result.append((m_gradient, m_gradient_new))
            result.append((ms_gradient, ms_gradient_new))
        return result

    def _get_model_updates(self):
        timestep = self.params['adam.timestep']
        timestep_new = timestep + 1.0
        alpha = self.learning_rate * tensor.sqrt(1.0 - (self._gamma_ms ** timestep_new)) \
                / (1.0 - (self._gamma_m ** timestep_new))

        result = []
        for name, param in self.network.params.items():
            m_gradient = self.params[name + '.mean_gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            rms_gradient = tensor.sqrt(ms_gradient) + self._epsilon
            param_new = param - (alpha * m_gradient / rms_gradient)
            result.append((param, param_new))
        result.append((timestep, timestep_new))
        return result

    def reset(self):
        print("Resetting Adam timestep to zero.")
        self.params['adam.timestep'].set_value(numpy.float32(0.0))
