#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.optimizers.basicoptimizer import BasicOptimizer

class AdamOptimizer(BasicOptimizer):
    """Adam Optimization Method

    D. P. Kingma, J. Ba (2015)
    Adam: A Method for Stochastic Optimization
    The International Conference on Learning Representations (ICLR), San Diego
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates an Adam optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self.param_init_values = dict()

        self.param_init_values['optimizer/timestep'] = \
            numpy.dtype(theano.config.floatX).type(0.0)

        for name, param in network.params.items():
            self.param_init_values[name + '_gradient'] = \
                numpy.zeros_like(param.get_value())
            self.param_init_values[name + '_mean_gradient'] = \
                numpy.zeros_like(param.get_value())
            self.param_init_values[name + '_mean_sqr_gradient'] = \
                numpy.zeros_like(param.get_value())

        # geometric rate for averaging gradients
        if not 'gradient_decay_rate' in optimization_options:
            raise ValueError("Gradient decay rate is not given in training "
                             "options.")
        self._gamma_m = optimization_options['gradient_decay_rate']

        # geometric rate for averaging squared gradients
        if not 'sqr_gradient_decay_rate' in optimization_options:
            raise ValueError("Squared gradient decay rate is not given in "
                             "optimization options.")
        self._gamma_ms = optimization_options['sqr_gradient_decay_rate']

        # momentum
        if not 'momentum' in optimization_options:
            raise ValueError("Momentum is not given in optimization options.")
        self._momentum = optimization_options['momentum']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _gradient_update_exprs(self):
        result = []
        for name, gradient_new in zip(self.network.params,
                                      self._gradient_exprs):
            gradient = self.params[name + '_gradient']
            m_gradient = self.params[name + '_mean_gradient']
            ms_gradient = self.params[name + '_mean_sqr_gradient']
            m_gradient_new = \
                self._gamma_m * m_gradient + \
                (1.0 - self._gamma_m) * gradient
            ms_gradient_new = \
                self._gamma_ms * ms_gradient + \
                (1.0 - self._gamma_ms) * tensor.sqr(gradient)
            result.append((gradient, gradient_new))
            result.append((m_gradient, m_gradient_new))
            result.append((ms_gradient, ms_gradient_new))
        return result

    def _model_update_exprs(self, alpha):
        timestep = self.params['optimizer/timestep']
        timestep_new = timestep + 1.0
        alpha *= tensor.sqrt(1.0 - (self._gamma_ms ** timestep_new))
        alpha /= 1.0 - (self._gamma_m ** timestep_new)

        updates = dict()
        for name, param in self.network.params.items():
            m_gradient = self.params[name + '_mean_gradient']
            ms_gradient = self.params[name + '_mean_sqr_gradient']
            rms_gradient = tensor.sqrt(ms_gradient) + self._epsilon
            updates[name] = -m_gradient / rms_gradient
        self._normalize(updates)

        result = []
        for name, param in self.network.params.items():
            update = updates[name]
            result.append((param, param + alpha * update))
        result.append((timestep, timestep_new))
        return result
